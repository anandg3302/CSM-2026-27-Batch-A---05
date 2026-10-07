import io
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, status
from PIL import Image

from app.core.config import settings
from app.schemas.media import MediaCaseMetadata
from app.schemas.report import ForensicReport
from app.services.reporting.report_service import (
    _load_case_readonly,
    generate_forensic_report,
    get_forensic_report,
)

PAGE_WIDTH = 612  # 8.5 in * 72 pt
PAGE_HEIGHT = 792  # 11 in * 72 pt
MARGIN_LEFT = 48
MARGIN_RIGHT = 48
MARGIN_TOP = 54
MARGIN_BOTTOM = 54


def _escape_pdf_text(text: str) -> str:
    """Escape a string for safe inclusion inside a PDF literal text string (...)."""
    sanitized = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", " ")
        .replace("\n", " ")
    )
    return "".join(ch if 32 <= ord(ch) <= 126 else "?" for ch in sanitized)


def _prepare_image_xobject(image_path: Path, max_dim: int = 220) -> Optional[Tuple[bytes, int, int]]:
    """
    Load an image in read-only mode, convert to RGB thumbnail in memory,
    and return (jpeg_bytes, width, height) without modifying the source file.
    """
    if not image_path.is_file():
        return None
    try:
        with Image.open(image_path) as img:
            rgb_copy = img.convert("RGB")
            rgb_copy.thumbnail((max_dim, max_dim))
            w, h = rgb_copy.size
            buf = io.BytesIO()
            rgb_copy.save(buf, format="JPEG", quality=88)
            return buf.getvalue(), max(1, w), max(1, h)
    except Exception:
        return None


class _PdfDocumentBuilder:
    """
    Lightweight, standards-compliant PDF 1.4 multi-page document builder
    supporting vector headers, wrapped text blocks, and DCTDecode image XObjects.
    """

    def __init__(self) -> None:
        self.pages_commands: List[List[str]] = [[]]
        self.pages_images: List[Dict[str, Tuple[bytes, int, int]]] = [{}]
        self.y_cursor: float = PAGE_HEIGHT - MARGIN_TOP
        self._img_counter: int = 0

    def _ensure_space(self, needed_height: float) -> None:
        if self.y_cursor - needed_height < MARGIN_BOTTOM:
            self.new_page()

    def new_page(self) -> None:
        self.pages_commands.append([])
        self.pages_images.append({})
        self.y_cursor = PAGE_HEIGHT - MARGIN_TOP

    def add_banner(self, title: str, subtitle: str) -> None:
        self._ensure_space(60)
        cmds = self.pages_commands[-1]
        box_h = 46
        box_y = self.y_cursor - box_h
        box_w = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
        cmds.append(f"0.09 0.16 0.28 rg {MARGIN_LEFT} {box_y:.1f} {box_w} {box_h} re f")
        cmds.append(
            f"BT /F2 15 Tf 1 1 1 rg {MARGIN_LEFT + 12} {self.y_cursor - 20:.1f} Td ({_escape_pdf_text(title)}) Tj ET"
        )
        cmds.append(
            f"BT /F1 10 Tf 0.85 0.90 0.98 rg {MARGIN_LEFT + 12} {self.y_cursor - 36:.1f} Td ({_escape_pdf_text(subtitle)}) Tj ET"
        )
        cmds.append("0 0 0 rg")
        self.y_cursor -= box_h + 14

    def add_section_header(self, title: str) -> None:
        self._ensure_space(34)
        self.y_cursor -= 6
        cmds = self.pages_commands[-1]
        box_w = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
        cmds.append(f"0.92 0.95 0.99 rg {MARGIN_LEFT} {self.y_cursor - 18:.1f} {box_w} 20 re f")
        cmds.append(
            f"BT /F2 11 Tf 0.08 0.18 0.34 rg {MARGIN_LEFT + 8} {self.y_cursor - 13:.1f} Td ({_escape_pdf_text(title)}) Tj ET"
        )
        cmds.append("0 0 0 rg")
        self.y_cursor -= 26

    def add_line(
        self,
        text: str,
        *,
        font: str = "F1",
        size: int = 9,
        indent: int = 0,
        wrap_width: int = 88,
    ) -> None:
        wrapped = textwrap.wrap(text, width=wrap_width) or [""]
        line_height = size + 4
        for line in wrapped:
            self._ensure_space(line_height)
            x = MARGIN_LEFT + indent
            self.pages_commands[-1].append(
                f"BT /{font} {size} Tf {x} {self.y_cursor:.1f} Td ({_escape_pdf_text(line)}) Tj ET"
            )
            self.y_cursor -= line_height

    def add_kv(self, label: str, value: str) -> None:
        self.add_line(f"{label}: {value}", font="F1", size=9, indent=4, wrap_width=86)

    def add_spacer(self, pts: float = 8.0) -> None:
        self.y_cursor -= pts

    def add_image_row(
        self,
        items: List[Tuple[str, Optional[Path]]],
    ) -> None:
        """Render up to two labeled image artifacts side-by-side on the current page."""
        loaded: List[Tuple[str, Optional[Tuple[bytes, int, int]]]] = [
            (caption, _prepare_image_xobject(path, max_dim=180) if path else None)
            for caption, path in items
        ]

        has_any_image = any(img_data is not None for _, img_data in loaded)
        if not has_any_image:
            for caption, _ in loaded:
                self.add_line(f"{caption}: Artifact not available.", font="F1", size=9, indent=6)
            return

        row_height = 155.0
        self._ensure_space(row_height)
        cmds = self.pages_commands[-1]
        imgs_dict = self.pages_images[-1]

        col_width = (PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / max(1, len(loaded))
        for idx, (caption, img_data) in enumerate(loaded):
            col_x = MARGIN_LEFT + idx * col_width + 6
            cmds.append(
                f"BT /F2 8 Tf {col_x:.1f} {self.y_cursor - 10:.1f} Td ({_escape_pdf_text(caption)}) Tj ET"
            )
            if img_data is None:
                cmds.append(
                    f"BT /F1 8 Tf {col_x:.1f} {self.y_cursor - 30:.1f} Td (Artifact not available.) Tj ET"
                )
            else:
                jpeg_bytes, px_w, px_h = img_data
                scale = min(160.0 / px_w, 120.0 / px_h)
                draw_w = px_w * scale
                draw_h = px_h * scale
                draw_y = self.y_cursor - 16 - draw_h
                self._img_counter += 1
                img_name = f"Im{self._img_counter}"
                imgs_dict[img_name] = (jpeg_bytes, px_w, px_h)
                cmds.append(
                    f"q {draw_w:.2f} 0 0 {draw_h:.2f} {col_x:.2f} {draw_y:.2f} cm /{img_name} Do Q"
                )

        self.y_cursor -= row_height

    def build_bytes(self) -> bytes:
        """Assemble all PDF objects, streams, xref table, and trailer into valid PDF 1.4 bytes."""
        objects: List[bytes] = []

        def add_obj(content: bytes) -> int:
            objects.append(content)
            return len(objects)

        # Object 1: Catalog (placeholder updated after Pages object created)
        add_obj(b"")
        # Object 2: Pages (placeholder updated after Page objects created)
        add_obj(b"")
        # Object 3: Font F1 (Helvetica)
        f1_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        # Object 4: Font F2 (Helvetica-Bold)
        f2_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        # Object 5: Font F3 (Courier)
        f3_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

        page_obj_ids: List[int] = []

        for page_idx, cmds in enumerate(self.pages_commands):
            imgs = self.pages_images[page_idx]
            xobj_refs: List[str] = []
            for img_name, (jpeg_bytes, px_w, px_h) in imgs.items():
                img_header = (
                    f"<< /Type /XObject /Subtype /Image /Width {px_w} /Height {px_h} "
                    f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
                    f"/Length {len(jpeg_bytes)} >>\nstream\n"
                ).encode("ascii")
                img_obj_id = add_obj(img_header + jpeg_bytes + b"\nendstream")
                xobj_refs.append(f"/{img_name} {img_obj_id} 0 R")

            stream_body = "\n".join(cmds).encode("ascii")
            stream_obj_id = add_obj(
                f"<< /Length {len(stream_body)} >>\nstream\n".encode("ascii")
                + stream_body
                + b"\nendstream"
            )

            xobj_dict = f"/XObject << {' '.join(xobj_refs)} >>" if xobj_refs else ""
            page_dict = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {f1_id} 0 R /F2 {f2_id} 0 R /F3 {f3_id} 0 R >> "
                f"{xobj_dict} >> /Contents {stream_obj_id} 0 R >>"
            ).encode("ascii")
            page_obj_ids.append(add_obj(page_dict))

        kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_ids)} >>".encode(
            "ascii"
        )

        out = io.BytesIO()
        out.write(b"%PDF-1.4\n")
        offsets: List[int] = []
        for idx, obj_bytes in enumerate(objects, start=1):
            offsets.append(out.tell())
            out.write(f"{idx} 0 obj\n".encode("ascii"))
            out.write(obj_bytes)
            out.write(b"\nendobj\n")

        xref_offset = out.tell()
        out.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        out.write(b"0000000000 65535 f \n")
        for off in offsets:
            out.write(f"{off:010d} 00000 n \n".encode("ascii"))

        out.write(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        return out.getvalue()


def _render_report_to_pdf_bytes(
    report: ForensicReport,
    case_data: MediaCaseMetadata,
) -> bytes:
    builder = _PdfDocumentBuilder()

    # PAGE 1: Cover Header & Core Metadata
    builder.add_banner("MediaTrace", "DIGITAL MEDIA FORENSIC REPORT")
    builder.add_kv("Case ID", report.case_id)
    builder.add_kv("Report ID", report.report_id)
    builder.add_kv("Report Version", report.report_version)
    builder.add_kv("Generated", report.generated_at)
    builder.add_kv("Original Filename", report.case.original_filename)
    builder.add_spacer(6)

    # Integrity Summary
    builder.add_section_header("Integrity")
    builder.add_kv("SHA-256", report.integrity.sha256)
    builder.add_kv("Hash Preserved", "Yes" if report.integrity.hash_preserved else "No")
    builder.add_line(report.integrity.explanation, indent=4)
    builder.add_line(report.integrity.scope_notice, indent=4)
    builder.add_spacer(4)

    # Assessment Summary
    builder.add_section_header("Assessment")
    builder.add_kv("Assessment Code", report.assessment.assessment)
    builder.add_kv("Assessment Label", report.assessment.assessment_label)
    builder.add_line(report.assessment.note, indent=4)
    builder.add_spacer(4)

    # SECTION 1: CASE INFORMATION
    builder.add_section_header("SECTION 1 - CASE INFORMATION")
    builder.add_kv("Case ID", report.case.case_id)
    builder.add_kv("Original Filename", report.case.original_filename)
    builder.add_kv("Media Type", report.case.media_type)
    builder.add_kv("File Extension", report.case.extension)
    builder.add_kv("File Size (Bytes)", str(report.case.file_size))
    builder.add_kv("Case Created At", report.case.created_at)
    builder.add_kv("Case Status", report.case.status)

    orig_img_path = settings.uploads_dir / case_data.stored_filename
    builder.add_image_row([("Original Image Preview", orig_img_path)])

    # SECTION 2: MEDIA & FILE FORENSICS
    builder.add_section_header("SECTION 2 - MEDIA & FILE FORENSICS")
    if report.media.available:
        builder.add_kv("Format", str(report.media.format or "N/A"))
        builder.add_kv("Dimensions", f"{report.media.width}x{report.media.height} px")
        builder.add_kv("Color Mode", str(report.media.color_mode or "N/A"))
        builder.add_kv("File Type", report.media.file_type)
        builder.add_kv("EXIF Present", "Yes" if report.media.exif_present else "No")
        builder.add_kv(
            "Software Metadata Present",
            "Yes" if report.media.software_metadata_present else "No",
        )
        builder.add_kv(
            "Camera Metadata Present",
            "Yes" if report.media.camera_metadata_present else "No",
        )
        builder.add_kv(
            "Timestamp Metadata Present",
            "Yes" if report.media.timestamp_metadata_present else "No",
        )
        builder.add_kv(
            "GPS Metadata Presence",
            report.media.gps_notice if report.media.gps_metadata_present else "Not present",
        )
        if report.forensics.software_metadata:
            builder.add_line(report.forensics.software_metadata, indent=4)
        if report.forensics.camera_information:
            builder.add_line(report.forensics.camera_information, indent=4)
        if report.forensics.timestamps:
            builder.add_line(report.forensics.timestamps, indent=4)
        if report.forensics.jpeg_properties:
            builder.add_line(report.forensics.jpeg_properties, indent=4)
    else:
        builder.add_line(report.forensics.exif_summary, indent=4)

    # SECTION 3: MANIPULATION DETECTION
    builder.add_section_header("SECTION 3 - MANIPULATION DETECTION")
    builder.add_kv("Detector Status", report.manipulation_detection.detector_status)
    if report.manipulation_detection.available:
        builder.add_kv(
            "Overall Assessment",
            f"{report.manipulation_detection.assessment_label} ({report.manipulation_detection.overall_assessment})",
        )
        builder.add_kv(
            "Recompression Status",
            str(report.manipulation_detection.recompression_status or "Not evaluated"),
        )
        for sig in report.manipulation_detection.forensic_signals:
            builder.add_line(
                f"- [{sig.strength}] {sig.name}: {sig.value} ({sig.description})",
                indent=6,
            )
    builder.add_line(report.manipulation_detection.ai_generation_label, font="F2", indent=4)
    builder.add_line(report.manipulation_detection.ai_generation_explanation, indent=4)
    ai_det_rep = report.ai_generation_detection
    if ai_det_rep is not None and ai_det_rep.status == "available" and ai_det_rep.prediction is not None:
        interp = ai_det_rep.ai_detector_interpretation
        builder.add_line("AI DETECTOR - MODEL OUTPUT (UNCALIBRATED):", font="F2", indent=4)
        builder.add_line(
            f"Selected Class: {interp.model_output.selected_class} (ID: {interp.model_output.selected_class_id}) | "
            f"Artificial Score: {interp.model_output.artificial_score} | "
            f"Human Score: {interp.model_output.human_score} | "
            f"Score Type: {interp.model_output.score_type} | "
            f"Decision Rule: {interp.model_output.decision_rule} | Threshold: null",
            indent=6,
        )
        builder.add_line("AI DETECTOR - FORENSIC INTERPRETATION:", font="F2", indent=4)
        builder.add_line(interp.forensic_meaning, indent=6)
        builder.add_line(f"Recommended Use: {interp.recommended_use}", indent=6)
        for p_inf in interp.prohibited_inference:
            builder.add_line(f"- Prohibited Inference: {p_inf}", indent=6)
        if ai_det_rep.transformation_context.documented:
            builder.add_line(
                f"Transformation Context: {', '.join(ai_det_rep.transformation_context.signals)} - {ai_det_rep.transformation_context.note}",
                indent=6,
            )
        builder.add_line("AI DETECTOR - LIMITATIONS:", font="F2", indent=4)
        for lim in ai_det_rep.limitations:
            builder.add_line(f"- {lim}", indent=6)


    # SECTION 4: LOCALIZATION
    builder.add_section_header("SECTION 4 - LOCALIZATION")
    builder.add_kv("Localization Status", report.localization.status)
    builder.add_kv(
        "Localized Forensic Anomaly",
        "Detected (localized forensic anomaly)"
        if report.localization.localized_anomaly
        else "Not detected",
    )
    if report.localization.regions:
        for reg in report.localization.regions:
            builder.add_line(
                f"- Region {reg.id} (localized forensic anomaly): x={reg.x}, y={reg.y}, "
                f"width={reg.width}, height={reg.height}, area={reg.area}, "
                f"mean_anomaly_score={reg.mean_anomaly_score}, max_anomaly_score={reg.max_anomaly_score}",
                indent=6,
            )
    heatmap_path = settings.results_dir / report.case_id / "localization" / "heatmap.png"
    mask_path = settings.results_dir / report.case_id / "localization" / "mask.png"
    builder.add_image_row(
        [
            ("Localization Heatmap", heatmap_path),
            ("Localization Mask", mask_path),
        ]
    )
    builder.add_line(report.localization.disclaimer, indent=4)

    # SECTION 5: REFERENCE IMAGE COMPARISON
    builder.add_section_header("SECTION 5 - REFERENCE IMAGE COMPARISON")
    if report.comparison.available:
        builder.add_kv("Reference Filename", str(report.comparison.reference_filename or "N/A"))
        builder.add_kv("Reference SHA-256", str(report.comparison.reference_sha256 or "N/A"))
        builder.add_kv(
            "Dimensions",
            f"Case {report.comparison.case_dimensions} vs Reference {report.comparison.reference_dimensions}",
        )
        builder.add_kv(
            "Changed Pixel Percentage",
            f"{report.comparison.changed_pixel_percentage:.2f}% (visual differences)",
        )
        builder.add_kv("Changed Region Count", str(report.comparison.changed_region_count))
        ref_preview_path = (
            settings.uploads_dir / case_data.reference_image.stored_filename
            if case_data.reference_image is not None
            else None
        )
        diff_img_path = settings.results_dir / report.case_id / "comparison" / "difference.png"
        diff_mask_path = (
            settings.results_dir / report.case_id / "comparison" / "difference_mask.png"
        )
        builder.add_image_row(
            [
                ("Reference Image Preview", ref_preview_path),
                ("Difference Image", diff_img_path),
            ]
        )
        builder.add_image_row([("Difference Mask", diff_mask_path)])
    else:
        builder.add_line("Reference image comparison was not executed for this case.", indent=4)
    builder.add_line(report.comparison.limitation_notice, indent=4)

    # SECTION 6: EVIDENCE CORRELATION
    builder.add_section_header("SECTION 6 - EVIDENCE CORRELATION")
    builder.add_kv("Total Evidence Items", str(report.evidence.summary.total_evidence_items))
    builder.add_kv("Informational Items", str(report.evidence.summary.informational))
    builder.add_kv("Weak Items", str(report.evidence.summary.weak))
    builder.add_kv("Moderate Items", str(report.evidence.summary.moderate))
    builder.add_kv(
        "Analysis Stages Available",
        ", ".join(report.evidence.summary.analysis_stages_available),
    )
    if report.evidence.cross_analysis_evidence:
        builder.add_line("Cross-analysis spatial consistency:", font="F2", indent=4)
        for cross in report.evidence.cross_analysis_evidence:
            builder.add_line(
                f"- {cross.id}: Independent analyses identify an overlapping region "
                f"(localization={cross.localization_region}, comparison={cross.comparison_region}, "
                f"overlap={cross.overlap_percentage}%, area={cross.intersection_area}, strength={cross.strength})",
                indent=6,
            )
    for item in report.evidence.items:
        builder.add_line(
            f"- {item.id} [{item.category}/{item.type}] ({item.strength}, source={item.source}): "
            f"{item.title} - {item.description}",
            indent=6,
        )

    # SECTION 6B: MULTI-SIGNAL FORENSIC INTERPRETATION (STEP 14.2)
    builder.add_section_header("MULTI-SIGNAL FORENSIC INTERPRETATION")
    ms = report.multi_signal_interpretation
    if ms is not None:
        builder.add_line("1. Signal Inventory", font="F2", indent=4)
        for sig in ms.signal_inventory:
            builder.add_line(
                f"- [{sig.signal_id}] {sig.signal_category} (availability={sig.availability}, "
                f"strength={sig.strength}, group={sig.independence_group}): {sig.observation}",
                indent=6,
            )
            builder.add_line(f"Interpretation: {sig.forensic_interpretation}", indent=8)

        builder.add_line("2. Evidence Independence", font="F2", indent=4)
        for grp in ms.independence_groups:
            builder.add_line(
                f"- {grp.group_label} ({grp.group_id}): available={grp.available}, "
                f"meaningful_observation={grp.meaningful_observation_present}, "
                f"signals={', '.join(grp.signal_ids) or 'none'}",
                indent=6,
            )
            builder.add_line(f"Independence Note: {grp.independence_note}", indent=8)

        builder.add_line("3. Signal Relationships", font="F2", indent=4)
        builder.add_kv("Overall Consistency Classification", ms.consistency_classification)
        for rel in ms.signal_relationships:
            builder.add_line(
                f"- {rel.signal_a} <-> {rel.signal_b}: relationship={rel.relationship} "
                f"(consistency={rel.consistency}) - {rel.reason}",
                indent=6,
            )

        builder.add_line("4. Overall Qualitative Assessment", font="F2", indent=4)
        builder.add_kv("Qualitative Assessment Code", ms.overall_assessment)
        builder.add_kv("Qualitative Assessment Label", ms.overall_assessment_label)
        builder.add_line(ms.interpretation, indent=6)
        builder.add_line(f"Transformation Context: {ms.transformation_context.statement}", indent=6)

        builder.add_line("5. Supported Conclusions", font="F2", indent=4)
        for sc in ms.supported_conclusions:
            builder.add_line(f"- {sc}", indent=6)

        builder.add_line("6. Unsupported Conclusions", font="F2", indent=4)
        for usc in ms.unsupported_conclusions:
            builder.add_line(f"- {usc}", indent=6)

        builder.add_line("7. Alternative Explanations", font="F2", indent=4)
        for alt in ms.alternative_explanations:
            builder.add_line(f"- {alt}", indent=6)

        builder.add_line("8. Additional Evidence Recommendations", font="F2", indent=4)
        for rec in ms.additional_evidence_recommendations:
            builder.add_line(f"- {rec}", indent=6)

        builder.add_line("9. Limitations", font="F2", indent=4)
        for lim in ms.limitations:
            builder.add_line(f"- {lim}", indent=6)
    else:
        builder.add_line("Multi-signal forensic interpretation has not been generated.", indent=4)

    # SECTION 7: POSSIBLE EDIT-HISTORY RECONSTRUCTION
    builder.add_section_header("SECTION 7 - POSSIBLE EDIT-HISTORY RECONSTRUCTION")
    builder.add_line(report.history.title, font="F2", indent=4)
    builder.add_line(report.history.sequence_label, font="F2", indent=4)
    builder.add_line(report.history.historical_accuracy_label, font="F2", indent=4)
    builder.add_line(report.history.timeline_disclaimer, indent=4)
    builder.add_line(report.history.reference_comparison_disclaimer, indent=4)
    for ev in report.history.events:
        status_badge = ev.status.upper().replace("_", " ")
        builder.add_line(
            f"#{ev.sequence} [{ev.id}] [{status_badge}] {ev.title} "
            f"(type={ev.event_type}, confidence={ev.confidence}, temporal={ev.temporal_certainty}, "
            f"evidence={', '.join(ev.evidence_ids)})",
            font="F2",
            indent=4,
        )
        builder.add_line(f"Reasoning: {ev.reasoning}", indent=8)
        for lim in ev.limitations:
            builder.add_line(f"Limitation: {lim}", indent=8)

    # SECTION 8: KNOWN FACTS
    builder.add_section_header("SECTION 8 - KNOWN FACTS")
    for kf in report.history.known_facts:
        builder.add_line(
            f"- {kf.fact} (source={kf.source}, evidence={', '.join(kf.evidence_ids)})",
            indent=6,
        )

    # SECTION 9: INFERRED EVENTS
    builder.add_section_header("SECTION 9 - INFERRED EVENTS")
    if report.history.inferred_events:
        for inf in report.history.inferred_events:
            builder.add_line(
                f"- {inf.title} [{inf.event_type}] (confidence={inf.confidence}, "
                f"evidence={', '.join(inf.evidence_ids)}): {inf.reasoning}",
                indent=6,
            )
    else:
        builder.add_line("No inferred events recorded beyond direct observations.", indent=6)

    # SECTION 10: UNKNOWN / UNDETERMINED ITEMS
    builder.add_section_header("SECTION 10 - UNKNOWN / UNDETERMINED ITEMS")
    for unk in report.history.unknowns:
        builder.add_line(f"- {unk}", indent=6)

    # SECTION 11: FORENSIC REPRODUCIBILITY & AUDIT TRAIL (Step 14.3)
    builder.add_section_header("FORENSIC REPRODUCIBILITY & AUDIT TRAIL")
    snap = report.reproducibility_audit_trail
    if snap is not None:
        builder.add_line("1. Snapshot Identity", font="F2", indent=4)
        builder.add_line(
            f"Snapshot ID: {snap.snapshot_id} | Schema: v{snap.snapshot_schema_version} | Created: {snap.created_at}",
            indent=8,
        )
        builder.add_line(f"Canonical Snapshot SHA-256: {snap.snapshot_sha256}", indent=8)

        builder.add_line("2. Input File Provenance & Integrity Verification", font="F2", indent=4)
        inp = snap.input_file_provenance
        integ = snap.input_integrity
        builder.add_line(
            f"Original File: {inp.original_filename} ({inp.byte_size} bytes, media_type={inp.media_type})",
            indent=8,
        )
        builder.add_line(f"Original Upload SHA-256: {inp.sha256}", indent=8)
        builder.add_line(
            f"Integrity Verification Status: {integ.status.upper()} (hash_match={integ.hash_match}, "
            f"recomputed_sha256={integ.recomputed_sha256 or 'unavailable'})",
            indent=8,
        )

        builder.add_line("3. Executed Analyses & Versions", font="F2", indent=4)
        env = snap.environment_info
        builder.add_line(
            f"Application: {env.application_name} v{env.application_version} | Device: {env.execution_device} | "
            f"Platform: {env.platform_category}",
            indent=8,
        )
        if snap.analyses_executed:
            for anl in snap.analyses_executed:
                rerun_note = f", rerun_of={anl.rerun_of_analysis_id}" if anl.rerun_of_analysis_id else ""
                reuse_note = f", reused={anl.reused_analysis_id}" if anl.reused_analysis_id else ""
                builder.add_line(
                    f"- [{anl.analysis_id}] {anl.service_name} v{anl.service_version} "
                    f"(status={anl.status}, config_hash={anl.configuration_hash[:12]}...{rerun_note}{reuse_note}): "
                    f"{anl.output_summary}",
                    indent=8,
                )
        else:
            builder.add_line(
                f"Historical Analysis Registry: {snap.reproducibility_check.historical_analysis_registry}",
                indent=8,
            )

        builder.add_line("4. AI Model Provenance", font="F2", indent=4)
        aip = snap.ai_model_provenance
        builder.add_line(
            f"Model: {aip.model_name} | Pinned Revision: {aip.pinned_revision} | "
            f"Architecture: {aip.architecture} | Status: {aip.status}",
            indent=8,
        )
        builder.add_line(
            f"Score Type: {aip.score_type} | Decision Rule: {aip.decision_rule} | Threshold: None | "
            f"ONNX SHA-256: {(aip.onnx_sha256 or 'N/A')[:16]}...",
            indent=8,
        )

        builder.add_line("5. Derived Artifacts & Hashes", font="F2", indent=4)
        if snap.artifacts_produced:
            for art in snap.artifacts_produced:
                sup_note = f" [superseded by {art.superseded_by_artifact_id}]" if art.superseded_by_artifact_id else ""
                builder.add_line(
                    f"- [{art.artifact_id}] {art.artifact_type}: {art.relative_path} "
                    f"(SHA-256={art.sha256[:16]}..., {art.byte_size} bytes){sup_note}",
                    indent=8,
                )
        else:
            builder.add_line("No derived visualization or report artifacts registered yet.", indent=8)

        builder.add_line("6. Evidence Dependency Chain", font="F2", indent=4)
        builder.add_line(
            f"Total Dependency Graph Nodes: {len(snap.evidence_dependency_graph)} | "
            f"Consumed Evidence IDs: {', '.join(snap.evidence_items_consumed) if snap.evidence_items_consumed else 'None'}",
            indent=8,
        )

        builder.add_line("7. Reproducibility Check Result", font="F2", indent=4)
        rc = snap.reproducibility_check
        builder.add_line(
            f"Reproducibility Status: {rc.status.upper()} (input_sha256_match={rc.input_sha256_match}, "
            f"config_hash_match={rc.configuration_hash_match}, artifact_hashes_match={rc.artifact_hashes_match})",
            indent=8,
        )
        for r_reason in rc.reasons:
            builder.add_line(f"  * Reason: {r_reason}", indent=10)

        builder.add_line("8. System Audit Events (Separate from Media Edit History)", font="F2", indent=4)
        if report.audit_events:
            for ev in report.audit_events[-12:]:
                builder.add_line(
                    f"- [{ev.event_id}] {ev.timestamp} | {ev.event_type} ({ev.event_category}): {ev.summary}",
                    indent=8,
                )
        else:
            builder.add_line("No system audit events recorded.", indent=8)

        builder.add_line("9. Reproducibility Limitations", font="F2", indent=4)
        for r_lim in snap.reproducibility_limitations:
            builder.add_line(f"- {r_lim}", indent=8)

    # SECTION 12: LIMITATIONS
    builder.add_section_header("SECTION 12 - LIMITATIONS")
    for lim in report.limitations:
        builder.add_line(f"- {lim}", indent=6)

    # FINAL NOTICE
    builder.add_section_header("FINAL NOTICE")
    builder.add_line(
        "MediaTrace provides forensic observations and evidence correlation. Findings should be "
        "interpreted in context and do not by themselves establish intent, authorship, or a "
        "complete historical editing sequence.",
        font="F2",
        indent=4,
    )
    builder.add_line(report.final_notice, indent=4)

    return builder.build_bytes()


def generate_pdf_report_for_case(case_id: str) -> Path:
    """
    Verify the case and structured ForensicReport JSON exist (generating report.json if not yet present),
    render the PDF document, and save it under storage/reports/<case_id>/MediaTrace_Forensic_Report.pdf.
    """
    case_data = _load_case_readonly(case_id)
    report_json_path = settings.reports_dir / case_id / "report.json"
    if report_json_path.is_file():
        report = get_forensic_report(case_id)
    else:
        report = generate_forensic_report(case_id)

    pdf_bytes = _render_report_to_pdf_bytes(report, case_data)

    case_report_dir = settings.reports_dir / case_id
    case_report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = case_report_dir / "MediaTrace_Forensic_Report.pdf"
    temp_path = pdf_path.with_suffix(".pdf.tmp")

    try:
        with open(temp_path, "wb") as file_out:
            file_out.write(pdf_bytes)
        os.replace(temp_path, pdf_path)
    except Exception as exc:
        if temp_path.is_file():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to write PDF forensic report.",
        ) from exc

    return pdf_path


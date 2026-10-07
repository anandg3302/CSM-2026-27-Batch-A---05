from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import zipfile

from fastapi import HTTPException, status
from PIL import Image, ImageDraw
import pypdf

from app.core.config import PROJECT_ROOT, settings
from app.schemas.document_analysis import (
    AllowedDocumentAssessment,
    DocumentAiDetectionResult,
    DocumentAnalysisResult,
    DocumentArtifactItem,
    DocumentConsistencyAnalysis,
    DocumentConsistencySignal,
    DocumentEmbeddedContent,
    DocumentForensicAssessment,
    DocumentMetadataInfo,
    DocumentParserStatus,
    DocumentRevisionAnalysis,
    DocumentStructureInfo,
    DocumentTextAnalysis,
    DocumentTimestampAnalysis,
    EmbeddedContentBucket,
    PageDimensionItem,
    TimestampRelationshipSignal,
)
from app.schemas.media import MediaCaseMetadata
from app.services import media_service

MAX_ZIP_UNCOMPRESSED_BYTES = 100 * 1024 * 1024  # 100 MB zip-bomb ceiling
MAX_ZIP_ENTRIES = 2000
ALLOWED_DOC_ARTIFACT_PATTERN = re.compile(r"^page_\d{3}\.png$")

MIME_TYPE_MAP = {
    ".pdf": ("PDF", "application/pdf"),
    ".docx": (
        "DOCX",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".doc": ("DOC", "application/msword"),
    ".txt": ("TXT", "text/plain"),
    ".rtf": ("RTF", "application/rtf"),
    ".odt": ("ODT", "application/vnd.oasis.opendocument.text"),
}

STANDARD_DOCUMENT_LIMITATIONS = [
    "Document metadata does not prove authorship and can be modified by standard office or PDF tools.",
    "Document metadata does not independently establish authenticity.",
    "Creation and modification timestamps may be rewritten by editors, operating systems, or export pipelines.",
    "File copying, downloading, or format conversion can alter filesystem and embedded timestamps.",
    "Revision indicators or incremental updates do not provide a complete historical editing sequence.",
    "Structural anomalies or multiple producers may result from normal document-generation or conversion software.",
    "Embedded objects, images, or hyperlinks do not by themselves imply malicious activity or tampering.",
    "AI-generated document detection is unavailable without a trained detection model.",
    "Exact chronological document history cannot be reconstructed from the file alone.",
    "Unsupported or legacy binary formats (such as legacy .doc) have limited structural inspection.",
]


def _compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _detect_gps_in_text(text: str) -> bool:
    lowered = text.lower()
    if any(
        k in lowered
        for k in (
            "gps",
            "location",
            "latitude",
            "longitude",
            "geo:",
            "exif:gps",
        )
    ):
        return True
    if re.search(r"[+-]\d{2}\.\d{3,}[,\s]+[+-]?\d{2,3}\.\d{3,}", text):
        return True
    return False


def _sanitize_meta_str(val: Any) -> Tuple[Optional[str], bool]:
    """Return (sanitized_value, gps_detected). Never returns raw GPS coordinates."""
    if val is None:
        return None, False
    text = str(val).strip()
    if not text:
        return None, False
    if _detect_gps_in_text(text):
        return "[Redacted Location Metadata]", True
    return text[:280], False


def _safe_parse_xml(xml_bytes: bytes) -> ET.Element:
    """
    Parse XML safely while blocking DOCTYPE / ENTITY declarations to prevent XXE
    and entity-expansion attacks.
    """
    head_upper = xml_bytes[:4096].upper()
    if b"<!DOCTYPE" in head_upper or b"<!ENTITY" in head_upper:
        raise ValueError("Unsafe XML DOCTYPE or ENTITY declaration rejected.")
    return ET.fromstring(xml_bytes)


def _open_safe_zip(file_path: Path) -> zipfile.ZipFile:
    """
    Open a ZIP-based document container (.docx / .odt) and verify it is not a zip bomb.
    """
    try:
        zf = zipfile.ZipFile(file_path, "r")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed or corrupt archive package in '{file_path.name}'.",
        ) from exc

    infos = zf.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        zf.close()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document archive exceeds maximum permitted entry count.",
        )

    total_uncompressed = sum(max(0, info.file_size) for info in infos)
    if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
        zf.close()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document archive uncompressed size exceeds safety limits.",
        )
    return zf


def _parse_iso_or_pdf_timestamp(raw_ts: Optional[str]) -> Optional[datetime]:
    if not raw_ts:
        return None
    s = raw_ts.strip()
    # Handle PDF D:YYYYMMDDHHMMSS format
    if s.startswith("D:"):
        s = s[2:]
    m_pdf = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", s)
    if m_pdf:
        try:
            year = int(m_pdf.group(1))
            month = int(m_pdf.group(2))
            day = int(m_pdf.group(3))
            hour = int(m_pdf.group(4) or 0)
            minute = int(m_pdf.group(5) or 0)
            sec = int(m_pdf.group(6) or 0)
            return datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc)
        except Exception:
            pass

    try:
        normalized = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _compute_text_statistics(
    full_text: str,
    *,
    encoding: Optional[str] = "UTF-8",
    bom_present: Optional[bool] = False,
    newline_style: Optional[str] = None,
    page_count: Optional[int] = None,
    paragraph_count_override: Optional[int] = None,
) -> DocumentTextAnalysis:
    max_chars = int(settings.max_extracted_text_chars)
    truncated = False
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
        truncated = True

    char_count = len(full_text)
    words = re.findall(r"\S+", full_text)
    word_count = len(words)
    lines = full_text.splitlines() if full_text else []
    line_count = len(lines)

    if paragraph_count_override is not None:
        paragraph_count = paragraph_count_override
    else:
        paragraphs = [p for p in re.split(r"\n\s*\n", full_text.strip()) if p.strip()]
        paragraph_count = len(paragraphs) if full_text.strip() else 0

    # Check repeated-text indicator (e.g. identical non-trivial line repeated >= 5 times)
    non_empty_lines = [ln.strip() for ln in lines if len(ln.strip()) > 20]
    repeated_indicator = False
    if non_empty_lines:
        counts: Dict[str, int] = {}
        for ln in non_empty_lines[:2000]:
            counts[ln] = counts.get(ln, 0) + 1
            if counts[ln] >= 5:
                repeated_indicator = True
                break

    unusual_encoding = "\ufffd" in full_text

    return DocumentTextAnalysis(
        status="truncated" if truncated else "available",
        encoding=encoding,
        bom_present=bom_present,
        newline_style=newline_style,
        character_count=char_count,
        word_count=word_count,
        line_count=line_count,
        paragraph_count=paragraph_count,
        page_count=page_count,
        language_hint="en" if re.search(r"\b(the|and|for|with|that)\b", full_text.lower()) else None,
        repeated_text_indicator=repeated_indicator,
        unusual_encoding_indicator=unusual_encoding,
        truncated=truncated,
    )


def _render_pdf_page_preview_artifact(
    page_number: int,
    width_pt: float,
    height_pt: float,
    page_text: str,
    image_count: int,
    annot_count: int,
    output_path: Path,
) -> None:
    """
    Generate a safe derived visual page layout preview card (page_001.png, etc.)
    under storage/results/<case_id>/document/ without executing any embedded PDF instructions.
    """
    canvas_w, canvas_h = 420, 560
    img = Image.new("RGB", (canvas_w, canvas_h), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    # Outer border & header bar
    draw.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=(148, 163, 184), width=2)
    draw.rectangle([0, 0, canvas_w - 1, 44], fill=(15, 23, 42))
    draw.text(
        (14, 14),
        f"PDF Page {page_number} Preview ({int(width_pt)}x{int(height_pt)} pt)",
        fill=(241, 245, 249),
    )
    draw.text(
        (14, 54),
        f"Images: {image_count} | Annotations/Links: {annot_count}",
        fill=(71, 85, 105),
    )
    draw.line([(14, 76), (canvas_w - 14, 76)], fill=(203, 213, 225), width=1)

    # Render sanitized text excerpt lines
    snippet_lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()][:18]
    y_cursor = 88
    if snippet_lines:
        for ln in snippet_lines:
            safe_ln = re.sub(r"[^\x20-\x7E]", " ", ln)[:52]
            draw.text((16, y_cursor), safe_ln, fill=(30, 41, 59))
            y_cursor += 22
            if y_cursor > canvas_h - 36:
                break
    else:
        draw.text((16, 100), "[No extractable text on this page]", fill=(100, 116, 139))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def _analyze_pdf_document(
    file_path: Path, case_id: str
) -> Tuple[
    DocumentMetadataInfo,
    DocumentStructureInfo,
    DocumentTextAnalysis,
    DocumentEmbeddedContent,
    DocumentRevisionAnalysis,
    List[DocumentArtifactItem],
]:
    raw_bytes = file_path.read_bytes()
    if not raw_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File does not begin with a valid %PDF- header.",
        )

    m_ver = re.match(rb"^%PDF-(\d+\.\d+)", raw_bytes[:20])
    pdf_version = m_ver.group(1).decode("ascii") if m_ver else "1.4"

    eof_count = raw_bytes.count(b"%%EOF")
    linearized = b"/Linearized" in raw_bytes[:2048]
    js_present = (b"/JavaScript" in raw_bytes) or (b"/JS" in raw_bytes)
    embedded_files_flag = b"/EmbeddedFiles" in raw_bytes
    forms_flag = b"/AcroForm" in raw_bytes

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes), strict=False)
        encrypted = bool(reader.is_encrypted)
        page_count = len(reader.pages) if not encrypted else 0
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Corrupt or unreadable PDF structure: {exc}",
        ) from exc

    meta = DocumentMetadataInfo(pdf_version=pdf_version)
    gps_found = _detect_gps_in_text(raw_bytes[:8192].decode("latin-1", errors="ignore"))

    if not encrypted and reader.metadata:
        raw_meta = reader.metadata
        for attr, key in (
            ("title", "/Title"),
            ("author", "/Author"),
            ("creator", "/Creator"),
            ("producer", "/Producer"),
            ("subject", "/Subject"),
            ("keywords", "/Keywords"),
            ("creation_timestamp", "/CreationDate"),
            ("modification_timestamp", "/ModDate"),
        ):
            val, g_flag = _sanitize_meta_str(raw_meta.get(key))
            if g_flag:
                gps_found = True
            setattr(meta, attr, val)

    meta.gps_present = gps_found
    meta.gps_notice = "GPS metadata present" if gps_found else None

    max_items = int(settings.max_embedded_items)
    max_preview_pages = int(settings.max_document_preview_pages)

    page_dims: List[PageDimensionItem] = []
    fonts_found: set[str] = set()
    image_items: List[Dict[str, Any]] = []
    hyperlink_items: List[Dict[str, Any]] = []
    annotation_items: List[Dict[str, Any]] = []
    page_texts: List[str] = []
    artifacts: List[DocumentArtifactItem] = []

    artifact_dir = settings.results_dir / case_id / "document"

    for idx, page in enumerate(reader.pages):
        p_num = idx + 1
        try:
            mb = page.mediabox
            w_pt = round(float(mb.width), 2)
            h_pt = round(float(mb.height), 2)
        except Exception:
            w_pt, h_pt = 612.0, 792.0

        if len(page_dims) < max_items:
            page_dims.append(PageDimensionItem(page_number=p_num, width_pt=w_pt, height_pt=h_pt))

        try:
            p_text = page.extract_text() or ""
        except Exception:
            p_text = ""
        page_texts.append(p_text)

        # Inspect resources for fonts and XObject images safely
        page_img_count = 0
        try:
            res = page.get("/Resources")
            if res:
                res_obj = res.get_object()
                font_dict = res_obj.get("/Font")
                if font_dict:
                    f_obj = font_dict.get_object()
                    for f_key in f_obj:
                        try:
                            f_entry = f_obj[f_key].get_object()
                            base_font = str(f_entry.get("/BaseFont", f_key)).lstrip("/")
                            fonts_found.add(base_font)
                        except Exception:
                            fonts_found.add(str(f_key).lstrip("/"))
                xobj_dict = res_obj.get("/XObject")
                if xobj_dict:
                    x_obj = xobj_dict.get_object()
                    for x_key in x_obj:
                        try:
                            item_obj = x_obj[x_key].get_object()
                            if str(item_obj.get("/Subtype")) == "/Image":
                                page_img_count += 1
                                if len(image_items) < max_items:
                                    image_items.append(
                                        {
                                            "page": p_num,
                                            "name": str(x_key).lstrip("/"),
                                            "width": int(item_obj.get("/Width", 0) or 0),
                                            "height": int(item_obj.get("/Height", 0) or 0),
                                        }
                                    )
                        except Exception:
                            pass
        except Exception:
            pass

        # Inspect annotations & hyperlinks safely
        page_annot_count = 0
        try:
            annots = page.get("/Annots")
            if annots:
                ann_list = annots.get_object()
                for ann in ann_list:
                    page_annot_count += 1
                    try:
                        a_obj = ann.get_object()
                        subtype = str(a_obj.get("/Subtype", "")).lstrip("/")
                        if len(annotation_items) < max_items:
                            annotation_items.append({"page": p_num, "subtype": subtype})
                        action = a_obj.get("/A")
                        if action:
                            act_obj = action.get_object()
                            uri = act_obj.get("/URI")
                            if uri and len(hyperlink_items) < max_items:
                                hyperlink_items.append({"page": p_num, "uri": str(uri)[:200]})
                    except Exception:
                        pass
        except Exception:
            pass

        if p_num <= max_preview_pages:
            art_name = f"page_{p_num:03d}.png"
            art_path = artifact_dir / art_name
            _render_pdf_page_preview_artifact(
                page_number=p_num,
                width_pt=w_pt,
                height_pt=h_pt,
                page_text=p_text,
                image_count=page_img_count,
                annot_count=page_annot_count,
                output_path=art_path,
            )
            artifacts.append(
                DocumentArtifactItem(
                    page_number=p_num,
                    artifact=f"storage/results/{case_id}/document/{art_name}",
                    artifact_filename=art_name,
                )
            )

    # Fallback regex check for raw /Subtype /Image and /URI in synthetic PDFs
    if not image_items and b"/Subtype /Image" in raw_bytes:
        raw_img_count = len(re.findall(rb"/Subtype\s*/Image", raw_bytes))
        for i in range(min(raw_img_count, max_items)):
            image_items.append({"page": 1, "name": f"Im{i+1}"})

    if not hyperlink_items and b"/URI" in raw_bytes:
        for m_uri in re.finditer(rb"/URI\s*\(([^)]+)\)", raw_bytes):
            if len(hyperlink_items) < max_items:
                hyperlink_items.append(
                    {"page": 1, "uri": m_uri.group(1).decode("latin-1", errors="ignore")[:200]}
                )

    full_text = "\n\n".join(page_texts)
    text_analysis = _compute_text_statistics(full_text, page_count=page_count)

    structure = DocumentStructureInfo(
        status="truncated" if page_count > max_preview_pages else "available",
        format_version=f"PDF-{pdf_version}",
        page_count=page_count,
        page_dimensions=page_dims,
        encrypted=encrypted,
        linearized=linearized,
        javascript_present=js_present,
        macro_present=False,
        object_count=len(re.findall(rb"\d+\s+\d+\s+obj\b", raw_bytes)),
        font_count=len(fonts_found),
        fonts=sorted(fonts_found)[:max_items],
        truncated=page_count > max_preview_pages,
    )

    embedded_content = DocumentEmbeddedContent(
        images=EmbeddedContentBucket(count=len(image_items), items=image_items),
        embedded_files=EmbeddedContentBucket(
            count=1 if embedded_files_flag else 0,
            items=[{"type": "EmbeddedFiles name tree"}] if embedded_files_flag else [],
        ),
        hyperlinks=EmbeddedContentBucket(count=len(hyperlink_items), items=hyperlink_items),
        annotations=EmbeddedContentBucket(count=len(annotation_items), items=annotation_items),
        forms=EmbeddedContentBucket(
            count=1 if forms_flag else 0,
            items=[{"type": "AcroForm"}] if forms_flag else [],
        ),
        objects=EmbeddedContentBucket(count=structure.object_count or 0, items=[]),
    )

    rev_indicators: List[str] = []
    if eof_count > 1:
        rev_indicators.append(
            f"Multiple %%EOF markers detected ({eof_count}), indicating possible incremental PDF saves/updates"
        )
    if meta.creation_timestamp and meta.modification_timestamp and meta.creation_timestamp != meta.modification_timestamp:
        rev_indicators.append("Modification timestamp differs from creation timestamp")

    revision_analysis = DocumentRevisionAnalysis(
        status="indicators_present" if rev_indicators else "no_strong_indicators",
        description=(
            "The document contains revision-related metadata or structural indicators that may warrant further review."
            if rev_indicators
            else "No incremental update markers or revision metadata indicators were detected."
        ),
        incremental_updates_detected=eof_count > 1,
        incremental_eof_count=eof_count,
        indicators=rev_indicators,
    )

    return meta, structure, text_analysis, embedded_content, revision_analysis, artifacts


def _analyze_docx_document(
    file_path: Path,
) -> Tuple[
    DocumentMetadataInfo,
    DocumentStructureInfo,
    DocumentTextAnalysis,
    DocumentEmbeddedContent,
    DocumentRevisionAnalysis,
]:
    zf = _open_safe_zip(file_path)
    try:
        names = zf.namelist()
        if "word/document.xml" not in names:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Archive does not contain required Office Open XML component 'word/document.xml'.",
            )

        meta = DocumentMetadataInfo(document_version="OOXML (DOCX)")
        gps_found = False

        # 1. Parse docProps/core.xml
        if "docProps/core.xml" in names:
            core_root = _safe_parse_xml(zf.read("docProps/core.xml"))
            for elem in core_root.iter():
                tag = elem.tag.split("}")[-1]
                val, g_flag = _sanitize_meta_str(elem.text)
                if g_flag:
                    gps_found = True
                if not val:
                    continue
                if tag == "title":
                    meta.title = val
                elif tag == "creator":
                    meta.author = val
                    meta.creator = val
                elif tag == "lastModifiedBy":
                    meta.last_modified_by = val
                elif tag == "subject":
                    meta.subject = val
                elif tag == "keywords":
                    meta.keywords = val
                elif tag == "revision":
                    meta.revision_number = val
                elif tag == "created":
                    meta.creation_timestamp = val
                elif tag == "modified":
                    meta.modification_timestamp = val

        # 2. Parse docProps/app.xml
        page_count: Optional[int] = None
        if "docProps/app.xml" in names:
            app_root = _safe_parse_xml(zf.read("docProps/app.xml"))
            for elem in app_root.iter():
                tag = elem.tag.split("}")[-1]
                val, g_flag = _sanitize_meta_str(elem.text)
                if g_flag:
                    gps_found = True
                if not val:
                    continue
                if tag == "Application":
                    meta.application = val
                elif tag == "Company":
                    meta.company = val
                elif tag == "Pages" and val.isdigit():
                    page_count = int(val)

        meta.gps_present = gps_found
        meta.gps_notice = "GPS metadata present" if gps_found else None

        # 3. Parse word/document.xml for text, paragraphs, tables, and tracked changes
        doc_root = _safe_parse_xml(zf.read("word/document.xml"))
        paragraphs: List[str] = []
        table_count = 0
        ins_count = 0
        del_count = 0
        fonts_found: set[str] = set()

        for elem in doc_root.iter():
            tag = elem.tag.split("}")[-1]
            if tag == "p":
                p_texts = [t.text for t in elem.iter() if t.tag.split("}")[-1] in ("t", "delText") and t.text]
                paragraphs.append("".join(p_texts))
            elif tag == "tbl":
                table_count += 1
            elif tag in ("ins", "moveTo"):
                ins_count += 1
            elif tag in ("del", "moveFrom"):
                del_count += 1
            elif tag == "rFonts":
                for attr_v in elem.attrib.values():
                    if attr_v:
                        fonts_found.add(attr_v)

        # Comments count
        comments_count = 0
        if "word/comments.xml" in names:
            comm_root = _safe_parse_xml(zf.read("word/comments.xml"))
            comments_count = sum(1 for e in comm_root.iter() if e.tag.split("}")[-1] == "comment")

        # Relationships & hyperlinks
        hyperlink_items: List[Dict[str, Any]] = []
        rel_count = 0
        if "word/_rels/document.xml.rels" in names:
            rel_root = _safe_parse_xml(zf.read("word/_rels/document.xml.rels"))
            for elem in rel_root.iter():
                if elem.tag.split("}")[-1] == "Relationship":
                    rel_count += 1
                    r_type = elem.attrib.get("Type", "")
                    target = elem.attrib.get("Target", "")
                    if "hyperlink" in r_type.lower() and target:
                        hyperlink_items.append({"target": target[:200], "type": "hyperlink"})

        # Embedded images and objects
        image_items = [
            {"filename": n, "size_bytes": zf.getinfo(n).file_size}
            for n in names
            if n.startswith("word/media/")
        ]
        embedded_obj_items = [
            {"filename": n, "size_bytes": zf.getinfo(n).file_size}
            for n in names
            if n.startswith("word/embeddings/")
        ]
        macro_present = any("vbaproject.bin" in n.lower() for n in names)
        header_count = sum(1 for n in names if re.match(r"^word/header\d*\.xml$", n))
        footer_count = sum(1 for n in names if re.match(r"^word/footer\d*\.xml$", n))

        full_text = "\n\n".join(p for p in paragraphs if p)
        text_analysis = _compute_text_statistics(
            full_text,
            page_count=page_count,
            paragraph_count_override=len(paragraphs),
        )

        structure = DocumentStructureInfo(
            status="available",
            format_version="OOXML",
            page_count=page_count,
            encrypted=False,
            macro_present=macro_present,
            package_components=sorted(names)[:50],
            font_count=len(fonts_found),
            fonts=sorted(fonts_found)[:50],
            paragraph_count=len(paragraphs),
            table_count=table_count,
            header_count=header_count,
            footer_count=footer_count,
            relationship_count=rel_count,
        )

        embedded_content = DocumentEmbeddedContent(
            images=EmbeddedContentBucket(count=len(image_items), items=image_items),
            embedded_files=EmbeddedContentBucket(count=len(embedded_obj_items), items=embedded_obj_items),
            hyperlinks=EmbeddedContentBucket(count=len(hyperlink_items), items=hyperlink_items),
            annotations=EmbeddedContentBucket(count=comments_count, items=[]),
            objects=EmbeddedContentBucket(count=len(embedded_obj_items), items=embedded_obj_items),
        )

        rev_indicators: List[str] = []
        tracked_detected = (ins_count + del_count) > 0
        if tracked_detected:
            rev_indicators.append(
                f"Tracked changes detected ({ins_count} insertion(s), {del_count} deletion(s))"
            )
        if comments_count > 0:
            rev_indicators.append(f"Document contains {comments_count} review comment(s)")
        if meta.revision_number and meta.revision_number.isdigit() and int(meta.revision_number) > 1:
            rev_indicators.append(f"Core properties record revision number {meta.revision_number}")
        if meta.author and meta.last_modified_by and meta.author != meta.last_modified_by:
            rev_indicators.append(
                f"Last modified by ('{meta.last_modified_by}') differs from creator ('{meta.author}')"
            )

        revision_analysis = DocumentRevisionAnalysis(
            status="indicators_present" if rev_indicators else "no_strong_indicators",
            description=(
                "The document contains revision-related metadata or structural indicators that may warrant further review."
                if rev_indicators
                else "No tracked changes or multi-author revision indicators were detected."
            ),
            tracked_changes_detected=tracked_detected,
            tracked_insertions_count=ins_count,
            tracked_deletions_count=del_count,
            comments_count=comments_count,
            revision_number=meta.revision_number,
            last_modified_by=meta.last_modified_by,
            indicators=rev_indicators,
        )

        return meta, structure, text_analysis, embedded_content, revision_analysis
    finally:
        zf.close()


def _analyze_odt_document(
    file_path: Path,
) -> Tuple[
    DocumentMetadataInfo,
    DocumentStructureInfo,
    DocumentTextAnalysis,
    DocumentEmbeddedContent,
    DocumentRevisionAnalysis,
]:
    zf = _open_safe_zip(file_path)
    try:
        names = zf.namelist()
        if "content.xml" not in names:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Archive does not contain required OpenDocument component 'content.xml'.",
            )

        meta = DocumentMetadataInfo(document_version="OpenDocument (ODT)")
        gps_found = False

        if "meta.xml" in names:
            meta_root = _safe_parse_xml(zf.read("meta.xml"))
            for elem in meta_root.iter():
                tag = elem.tag.split("}")[-1]
                val, g_flag = _sanitize_meta_str(elem.text)
                if g_flag:
                    gps_found = True
                if not val:
                    continue
                if tag == "title":
                    meta.title = val
                elif tag in ("initial-creator", "creator"):
                    if not meta.author:
                        meta.author = val
                    meta.last_modified_by = val
                elif tag == "creation-date":
                    meta.creation_timestamp = val
                elif tag == "date":
                    meta.modification_timestamp = val
                elif tag == "generator":
                    meta.generator = val
                elif tag == "editing-cycles":
                    meta.revision_number = val

        meta.gps_present = gps_found
        meta.gps_notice = "GPS metadata present" if gps_found else None

        content_root = _safe_parse_xml(zf.read("content.xml"))
        paragraphs: List[str] = []
        table_count = 0
        hyperlink_items: List[Dict[str, Any]] = []

        for elem in content_root.iter():
            tag = elem.tag.split("}")[-1]
            if tag == "p":
                txt = "".join(elem.itertext())
                if txt:
                    paragraphs.append(txt)
            elif tag == "table":
                table_count += 1
            elif tag == "a":
                for k, v in elem.attrib.items():
                    if k.endswith("href") and v:
                        hyperlink_items.append({"target": v[:200]})

        image_items = [
            {"filename": n, "size_bytes": zf.getinfo(n).file_size}
            for n in names
            if n.startswith("Pictures/")
        ]

        full_text = "\n\n".join(paragraphs)
        text_analysis = _compute_text_statistics(
            full_text,
            paragraph_count_override=len(paragraphs),
        )

        structure = DocumentStructureInfo(
            status="available",
            format_version="ODF 1.2/1.3",
            package_components=sorted(names)[:50],
            paragraph_count=len(paragraphs),
            table_count=table_count,
        )

        embedded_content = DocumentEmbeddedContent(
            images=EmbeddedContentBucket(count=len(image_items), items=image_items),
            hyperlinks=EmbeddedContentBucket(count=len(hyperlink_items), items=hyperlink_items),
        )

        rev_indicators: List[str] = []
        if meta.revision_number and meta.revision_number.isdigit() and int(meta.revision_number) > 1:
            rev_indicators.append(f"OpenDocument editing-cycles count is {meta.revision_number}")

        revision_analysis = DocumentRevisionAnalysis(
            status="indicators_present" if rev_indicators else "no_strong_indicators",
            description=(
                "The document contains revision-related metadata indicators."
                if rev_indicators
                else "No strong revision indicators were detected."
            ),
            revision_number=meta.revision_number,
            indicators=rev_indicators,
        )

        return meta, structure, text_analysis, embedded_content, revision_analysis
    finally:
        zf.close()


def _analyze_txt_or_rtf_document(
    file_path: Path, extension: str
) -> Tuple[
    DocumentMetadataInfo,
    DocumentStructureInfo,
    DocumentTextAnalysis,
    DocumentEmbeddedContent,
    DocumentRevisionAnalysis,
]:
    raw_bytes = file_path.read_bytes()
    bom_present = False
    encoding = "UTF-8"
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        bom_present = True
        encoding = "UTF-8-SIG"
        text_bytes = raw_bytes[3:]
    elif raw_bytes.startswith(b"\xff\xfe"):
        bom_present = True
        encoding = "UTF-16-LE"
        text_bytes = raw_bytes[2:]
    elif raw_bytes.startswith(b"\xfe\xff"):
        bom_present = True
        encoding = "UTF-16-BE"
        text_bytes = raw_bytes[2:]
    else:
        text_bytes = raw_bytes

    try:
        decoded = text_bytes.decode("utf-8" if "UTF-8" in encoding else encoding)
    except UnicodeDecodeError:
        encoding = "Latin-1"
        decoded = text_bytes.decode("latin-1", errors="replace")

    has_crlf = "\r\n" in decoded
    has_lf = "\n" in decoded.replace("\r\n", "")
    newline_style = "CRLF" if (has_crlf and not has_lf) else ("LF" if (has_lf and not has_crlf) else ("Mixed" if (has_crlf and has_lf) else "None"))

    meta = DocumentMetadataInfo()
    gps_found = _detect_gps_in_text(decoded[:16384])

    if extension == ".rtf":
        if not decoded.lstrip().startswith("{\\rtf"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File does not begin with a valid {\\rtf header.",
            )
        m_title = re.search(r"\{\\title\s+([^}]+)\}", decoded)
        m_author = re.search(r"\{\\author\s+([^}]+)\}", decoded)
        if m_title:
            meta.title, g1 = _sanitize_meta_str(m_title.group(1))
            gps_found = gps_found or g1
        if m_author:
            meta.author, g2 = _sanitize_meta_str(m_author.group(1))
            gps_found = gps_found or g2
        plain_text = re.sub(r"\\[a-zA-Z]+\d*\s?|[{}]", " ", decoded)
    else:
        plain_text = decoded

    meta.gps_present = gps_found
    meta.gps_notice = "GPS metadata present" if gps_found else None

    text_analysis = _compute_text_statistics(
        plain_text,
        encoding=encoding,
        bom_present=bom_present,
        newline_style=newline_style,
    )

    urls = re.findall(r"https?://[^\s)>\]\"']+", decoded)
    hyperlink_items = [{"uri": u[:200]} for u in urls[:50]]

    structure = DocumentStructureInfo(
        status="available",
        format_version="RTF 1.x" if extension == ".rtf" else f"Plain Text ({encoding})",
        paragraph_count=text_analysis.paragraph_count,
    )

    embedded_content = DocumentEmbeddedContent(
        hyperlinks=EmbeddedContentBucket(count=len(hyperlink_items), items=hyperlink_items),
    )

    revision_analysis = DocumentRevisionAnalysis(
        status="no_strong_indicators",
        description="No embedded revision history indicators detected.",
    )

    return meta, structure, text_analysis, embedded_content, revision_analysis


def analyze_document_for_case(case_id: str) -> MediaCaseMetadata:
    """
    Perform non-destructive forensic document analysis on an uploaded document case (Step 12).
    Verifies SHA-256 before and after analysis and stores any derived artifacts under
    storage/results/<case_id>/document/.
    """
    case = media_service.get_media_case(case_id)
    if case.media_type != "document":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Case '{case_id}' is of media_type '{case.media_type}'. Only document cases can be analyzed by the document analysis pipeline.",
        )

    doc_path = settings.uploads_dir / case.stored_filename
    if not doc_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stored document file for case '{case_id}' is missing on disk.",
        )

    # 1. Pre-analysis SHA-256 integrity verification
    pre_sha256 = _compute_file_sha256(doc_path)
    if pre_sha256 != case.sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="SHA-256 integrity mismatch detected on source document file before analysis.",
        )

    fmt_label, mime_type = MIME_TYPE_MAP.get(
        case.extension, (case.extension.lstrip(".").upper(), "application/octet-stream")
    )

    # 2. Handle legacy binary .doc conservatively as required by Section 8
    if case.extension == ".doc":
        legacy_reason = "Legacy DOC forensic parsing is not configured in the current environment."
        doc_result = DocumentAnalysisResult(
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            media_type="document",
            filename=case.original_filename,
            format="DOC",
            mime_type=mime_type,
            file_size=case.file_size,
            status="not_available",
            reason=legacy_reason,
            parser_status=DocumentParserStatus(status="not_available", reason=legacy_reason),
            structure=DocumentStructureInfo(status="not_available"),
            text_analysis=DocumentTextAnalysis(status="not_available"),
            revision_analysis=DocumentRevisionAnalysis(
                status="not_available",
                description=legacy_reason,
            ),
            consistency_analysis=DocumentConsistencyAnalysis(status="not_available"),
            ai_generation=DocumentAiDetectionResult(),
            assessment=DocumentForensicAssessment(
                status="insufficient_evidence",
                label="Legacy Format Parser Not Configured",
                description=legacy_reason,
            ),
            limitations=list(STANDARD_DOCUMENT_LIMITATIONS),
        )
        case.document_analysis = doc_result
        case.status = "analyzed"
        media_service.save_media_case(case)
        return case

    # 3. Run format-specific forensic inspection
    artifacts: List[DocumentArtifactItem] = []
    if case.extension == ".pdf":
        meta, structure, text_analysis, embedded_content, revision_analysis, artifacts = (
            _analyze_pdf_document(doc_path, case_id)
        )
    elif case.extension == ".docx":
        meta, structure, text_analysis, embedded_content, revision_analysis = (
            _analyze_docx_document(doc_path)
        )
    elif case.extension == ".odt":
        meta, structure, text_analysis, embedded_content, revision_analysis = (
            _analyze_odt_document(doc_path)
        )
    else:
        meta, structure, text_analysis, embedded_content, revision_analysis = (
            _analyze_txt_or_rtf_document(doc_path, case.extension)
        )

    # 4. Timestamp Consistency Analysis
    stat_info = doc_path.stat()
    fs_created = datetime.fromtimestamp(stat_info.st_ctime, tz=timezone.utc).isoformat()
    fs_modified = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc).isoformat()

    ts_signals: List[TimestampRelationshipSignal] = []
    consistency_signals: List[DocumentConsistencySignal] = []

    dt_created = _parse_iso_or_pdf_timestamp(meta.creation_timestamp)
    dt_modified = _parse_iso_or_pdf_timestamp(meta.modification_timestamp)

    if dt_created and dt_modified:
        if dt_created > dt_modified:
            msg = "Document creation timestamp occurs after document modification timestamp."
            ts_signals.append(
                TimestampRelationshipSignal(
                    type="timestamp_relationship",
                    status="potential_signal",
                    description=msg,
                )
            )
            consistency_signals.append(
                DocumentConsistencySignal(
                    finding="creation_after_modification_timestamp",
                    category="timestamp",
                    status="potential_signal",
                    description=msg,
                )
            )
        elif dt_created != dt_modified:
            ts_signals.append(
                TimestampRelationshipSignal(
                    type="timestamp_relationship",
                    status="informational",
                    description="Embedded modification timestamp differs from embedded creation timestamp.",
                )
            )

    if meta.modification_timestamp:
        ts_signals.append(
            TimestampRelationshipSignal(
                type="timestamp_relationship",
                status="informational",
                description="Embedded modification timestamp differs from the current filesystem modification timestamp.",
            )
        )

    timestamp_analysis = DocumentTimestampAnalysis(
        filesystem_created_at=fs_created,
        filesystem_modified_at=fs_modified,
        document_created_at=meta.creation_timestamp,
        document_modified_at=meta.modification_timestamp,
        signals=ts_signals,
    )

    # 5. Metadata & Structural Consistency Analysis
    if meta.creator and meta.producer and meta.creator.lower() != meta.producer.lower():
        consistency_signals.append(
            DocumentConsistencySignal(
                finding="creator_producer_difference",
                category="metadata",
                status="potential_signal",
                description=f"Document creator ('{meta.creator}') differs from producer ('{meta.producer}').",
            )
        )

    if meta.author and meta.last_modified_by and meta.author.lower() != meta.last_modified_by.lower():
        consistency_signals.append(
            DocumentConsistencySignal(
                finding="author_modifier_difference",
                category="metadata",
                status="potential_signal",
                description=f"Document author ('{meta.author}') differs from last_modified_by ('{meta.last_modified_by}').",
            )
        )

    if structure.javascript_present:
        consistency_signals.append(
            DocumentConsistencySignal(
                finding="embedded_pdf_javascript_present",
                category="structure",
                status="potential_signal",
                description="PDF contains embedded JavaScript actions (/JS or /JavaScript). Script was not executed.",
            )
        )

    if structure.macro_present:
        consistency_signals.append(
            DocumentConsistencySignal(
                finding="vba_macro_project_present",
                category="structure",
                status="potential_signal",
                description="Office package contains a VBA macro stream (vbaProject.bin). Macros were not executed.",
            )
        )

    consistency_analysis = DocumentConsistencyAnalysis(
        status="inconsistencies_detected" if consistency_signals else "consistent",
        signals=consistency_signals,
    )

    # 6. Conservative Qualitative Assessment
    has_struct_signal = bool(structure.javascript_present or structure.macro_present)
    has_meta_inconsistency = any(s.category in ("metadata", "timestamp") for s in consistency_signals)
    has_revisions = revision_analysis.status == "indicators_present"
    has_embedded = (
        embedded_content.images.count > 0
        or embedded_content.embedded_files.count > 0
        or embedded_content.objects.count > 0
    )

    if has_struct_signal:
        assessment = DocumentForensicAssessment(
            status="structural_inconsistency",
            label="Structural / Active-Content Signal Observed",
            description="Document contains embedded scripts, macros, or structural anomalies that warrant manual forensic inspection.",
        )
    elif has_meta_inconsistency:
        assessment = DocumentForensicAssessment(
            status="metadata_inconsistency",
            label="Potential Metadata / Timestamp Inconsistency Observed",
            description="Internal metadata or timestamp fields exhibit differences that may reflect editing, conversion, or multi-tool processing.",
        )
    elif has_revisions:
        assessment = DocumentForensicAssessment(
            status="revision_indicators_present",
            label="Revision / Incremental Update Indicators Present",
            description="The document contains tracked changes, comments, or incremental update markers indicating post-creation revisions.",
        )
    elif has_embedded:
        assessment = DocumentForensicAssessment(
            status="embedded_content_present",
            label="Embedded Content / Media Detected",
            description="Document contains embedded images, attachments, or objects with no strong structural or metadata anomalies.",
        )
    else:
        assessment = DocumentForensicAssessment(
            status="no_strong_indicators",
            label="No Strong Forensic Anomaly Indicators",
            description="No strong metadata conflicts, tracked revisions, or structural anomalies were observed.",
        )

    doc_result = DocumentAnalysisResult(
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        media_type="document",
        filename=case.original_filename,
        format=fmt_label,
        mime_type=mime_type,
        file_size=case.file_size,
        status="available",
        parser_status=DocumentParserStatus(status="available"),
        metadata=meta,
        structure=structure,
        text_analysis=text_analysis,
        embedded_content=embedded_content,
        revision_analysis=revision_analysis,
        timestamp_analysis=timestamp_analysis,
        consistency_analysis=consistency_analysis,
        artifacts=artifacts,
        ai_generation=DocumentAiDetectionResult(),
        assessment=assessment,
        limitations=list(STANDARD_DOCUMENT_LIMITATIONS),
    )

    # 7. Post-analysis SHA-256 verification to guarantee original document immutability
    post_sha256 = _compute_file_sha256(doc_path)
    if post_sha256 != case.sha256:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Post-analysis SHA-256 verification failed.",
        )

    case.document_analysis = doc_result
    case.status = "analyzed"

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        record_analysis_execution,
    )

    doc_art_specs = []
    doc_dir = settings.results_dir / case_id / "document"
    if doc_dir.is_dir():
        for png_file in sorted(doc_dir.glob("*.png")):
            doc_art_specs.append(
                {
                    "artifact_type": "document_page_preview",
                    "relative_path": f"storage/results/{case_id}/document/{png_file.name}",
                    "media_type": "image/png",
                }
            )

    record_analysis_execution(
        case_data=case,
        service_name="document_analysis_service",
        status="completed",
        input_sha256=case.sha256,
        output_summary=(
            f"Document analysis completed: format={fmt_label}, "
            f"pages={structure.page_count}, assessment={assessment.status}."
        ),
        artifact_specs=doc_art_specs,
        completed_at=doc_result.analyzed_at,
    )
    create_forensic_snapshot(case, record_event=True)

    return media_service.save_media_case(case)


def get_document_artifact_path(case_id: str, artifact_name: str) -> Path:
    """
    Validate and resolve a derived document preview artifact (e.g. page_001.png)
    under storage/results/<case_id>/document/. Prevents path traversal.
    """
    media_service.get_media_case(case_id)
    if not ALLOWED_DOC_ARTIFACT_PATTERN.match(artifact_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document artifact name '{artifact_name}'.",
        )

    artifact_dir = (settings.results_dir / case_id / "document").resolve()
    target_path = (artifact_dir / artifact_name).resolve()
    if artifact_dir not in target_path.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact path.",
        )

    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document artifact '{artifact_name}' not found for case '{case_id}'.",
        )

    return target_path

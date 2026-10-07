import io
import json
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient
from PIL import Image
import pypdf
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    StreamObject,
    TextStringObject,
)

from app.core.config import settings
from app.main import app

client = TestClient(app)


def create_synthetic_pdf_bytes(
    *,
    title: str = "Forensic Contract Draft",
    author: str = "Alice Investigator",
    creator: str = "Microsoft Word",
    producer: str = "Quartz PDFContext",
    creation_date: str = "D:20260928150000Z",
    mod_date: str = "D:20260928120000Z",  # Intentionally earlier than creation_date to test timestamp signal
    include_gps: bool = True,
    include_hyperlink: bool = True,
    include_image: bool = True,
) -> bytes:
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    # Add content stream with extractable text
    content_stream = StreamObject()
    content_stream._data = (
        b"BT /F1 12 Tf 72 720 Td (Forensic evidence document body text sample.) Tj ET"
    )
    content_ref = writer._add_object(content_stream)
    page[NameObject("/Contents")] = content_ref

    # Add Font and XObject Image resources
    font_dict = DictionaryObject(
        {
            NameObject("/F1"): DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
        }
    )
    resources = DictionaryObject({NameObject("/Font"): font_dict})

    if include_image:
        img_stream = StreamObject()
        img_stream._data = b"\xff\x00\xff"
        img_stream.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): NumberObject(32),
                NameObject("/Height"): NumberObject(32),
                NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                NameObject("/BitsPerComponent"): NumberObject(8),
            }
        )
        img_ref = writer._add_object(img_stream)
        resources[NameObject("/XObject")] = DictionaryObject({NameObject("/Im1"): img_ref})

    page[NameObject("/Resources")] = resources

    if include_hyperlink:
        annot = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Link"),
                NameObject("/Rect"): ArrayObject(
                    [NumberObject(72), NumberObject(700), NumberObject(200), NumberObject(725)]
                ),
                NameObject("/A"): DictionaryObject(
                    {
                        NameObject("/S"): NameObject("/URI"),
                        NameObject("/URI"): TextStringObject("https://example.org/evidence-ref"),
                    }
                ),
            }
        )
        annot_ref = writer._add_object(annot)
        page[NameObject("/Annots")] = ArrayObject([annot_ref])

    meta_dict = {
        "/Title": title,
        "/Author": author,
        "/Creator": creator,
        "/Producer": producer,
        "/CreationDate": creation_date,
        "/ModDate": mod_date,
    }
    if include_gps:
        meta_dict["/Keywords"] = "GPS Latitude 37.7749, -122.4194"
    writer.add_metadata(meta_dict)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_synthetic_docx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "docProps/core.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                               xmlns:dc="http://purl.org/dc/elements/1.1/"
                               xmlns:dcterms="http://purl.org/dc/terms/">
              <dc:title>Audit Statement</dc:title>
              <dc:creator>Origin Author</dc:creator>
              <cp:lastModifiedBy>Secondary Editor</cp:lastModifiedBy>
              <cp:revision>4</cp:revision>
              <dcterms:created>2026-09-20T10:00:00Z</dcterms:created>
              <dcterms:modified>2026-09-28T11:30:00Z</dcterms:modified>
            </cp:coreProperties>""",
        )
        zf.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
              <Application>Microsoft Office Word</Application>
              <Company>MediaTrace Lab</Company>
              <Pages>2</Pages>
            </Properties>""",
        )
        zf.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>First paragraph of the forensic DOCX evidence.</w:t></w:r></w:p>
                <w:p>
                  <w:ins w:id="1" w:author="Secondary Editor"><w:r><w:t>Inserted clause.</w:t></w:r></w:ins>
                  <w:del w:id="2" w:author="Secondary Editor"><w:r><w:delText>Removed clause.</w:delText></w:r></w:del>
                </w:p>
                <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Table Cell 1</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
              </w:body>
            </w:document>""",
        )
        zf.writestr(
            "word/comments.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:comment w:id="0" w:author="Reviewer"><w:p><w:r><w:t>Verify date</w:t></w:r></w:p></w:comment>
            </w:comments>""",
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/policy" TargetMode="External"/>
            </Relationships>""",
        )
        zf.writestr("word/media/image1.png", b"\x89PNG\r\n\x1a\nDUMMY_IMAGE")
    return buf.getvalue()


def create_synthetic_odt_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "meta.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
                                  xmlns:dc="http://purl.org/dc/elements/1.1/"
                                  xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0">
              <office:meta>
                <dc:title>OpenDocument Brief</dc:title>
                <meta:initial-creator>ODT Author</meta:initial-creator>
                <dc:creator>ODT Editor</dc:creator>
                <meta:generator>LibreOffice/7.6</meta:generator>
                <meta:editing-cycles>3</meta:editing-cycles>
              </office:meta>
            </office:document-meta>""",
        )
        zf.writestr(
            "content.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
                                     xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
                                     xmlns:xlink="http://www.w3.org/1999/xlink">
              <office:body>
                <office:text>
                  <text:p>OpenDocument body paragraph with <text:a xlink:href="https://example.org/odt">link</text:a>.</text:p>
                </office:text>
              </office:body>
            </office:document-content>""",
        )
        zf.writestr("Pictures/pic1.png", b"\x89PNG\r\n\x1a\nODT_IMG")
    return buf.getvalue()


def test_document_uploads_and_format_coverage_pdf_docx_txt_rtf_odt_doc():
    """
    Tests:
    - valid PDF, DOCX, TXT, RTF, ODT, and legacy DOC uploads and analysis
    - unsupported extension rejection
    - image, video, and audio cases sent to document endpoint rejected with 400
    """
    # 1. PDF upload & analysis
    pdf_up = client.post(
        "/api/media/upload",
        files={"file": ("contract_evidence.pdf", create_synthetic_pdf_bytes(), "application/pdf")},
    )
    assert pdf_up.status_code == 201
    pdf_case_id = pdf_up.json()["case_id"]
    pdf_path = Path(settings.storage_root) / "uploads" / pdf_up.json()["stored_filename"]
    pdf_bytes_before = pdf_path.read_bytes()

    pdf_res = client.post(f"/api/analysis/document/{pdf_case_id}")
    assert pdf_res.status_code == 200
    da = pdf_res.json()["document_analysis"]
    assert da["format"] == "PDF"
    assert da["structure"]["page_count"] == 1
    assert da["text_analysis"]["word_count"] >= 4
    assert da["embedded_content"]["images"]["count"] >= 1
    assert da["embedded_content"]["hyperlinks"]["count"] >= 1
    assert da["metadata"]["gps_present"] is True
    assert da["metadata"]["gps_notice"] == "GPS metadata present"
    assert "37.7749" not in json.dumps(da)
    assert da["consistency_analysis"]["status"] == "inconsistencies_detected"
    assert da["ai_generation"]["status"] == "not_available"
    assert "%" not in da["assessment"]["description"]
    assert len(da["limitations"]) >= 10
    assert pdf_path.read_bytes() == pdf_bytes_before

    # Verify bounded PDF page preview artifact
    assert len(da["artifacts"]) == 1
    art_resp = client.get(f"/api/analysis/document/{pdf_case_id}/artifacts/page_001.png")
    assert art_resp.status_code == 200
    assert art_resp.headers["content-type"] == "image/png"

    # 2. DOCX upload & analysis (core properties, paragraphs/tables, images, tracked changes)
    docx_up = client.post(
        "/api/media/upload",
        files={"file": ("statement.docx", create_synthetic_docx_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert docx_up.status_code == 201
    docx_res = client.post(f"/api/analysis/document/{docx_up.json()['case_id']}")
    assert docx_res.status_code == 200
    docx_da = docx_res.json()["document_analysis"]
    assert docx_da["metadata"]["title"] == "Audit Statement"
    assert docx_da["metadata"]["author"] == "Origin Author"
    assert docx_da["metadata"]["last_modified_by"] == "Secondary Editor"
    assert docx_da["structure"]["paragraph_count"] >= 2
    assert docx_da["structure"]["table_count"] == 1
    assert docx_da["embedded_content"]["images"]["count"] == 1
    assert docx_da["embedded_content"]["hyperlinks"]["count"] == 1
    assert docx_da["revision_analysis"]["tracked_changes_detected"] is True
    assert docx_da["revision_analysis"]["tracked_insertions_count"] == 1
    assert docx_da["revision_analysis"]["tracked_deletions_count"] == 1
    assert docx_da["revision_analysis"]["comments_count"] == 1

    # 3. TXT upload & analysis
    txt_up = client.post(
        "/api/media/upload",
        files={"file": ("notes.txt", b"\xef\xbb\xbfLine one\r\nLine two https://example.org\r\n", "text/plain")},
    )
    assert txt_up.status_code == 201
    txt_da = client.post(f"/api/analysis/document/{txt_up.json()['case_id']}").json()["document_analysis"]
    assert txt_da["text_analysis"]["bom_present"] is True
    assert txt_da["text_analysis"]["newline_style"] == "CRLF"
    assert txt_da["text_analysis"]["line_count"] == 2

    # 4. RTF upload & analysis
    rtf_up = client.post(
        "/api/media/upload",
        files={"file": ("memo.rtf", b"{\\rtf1\\ansi{\\info{\\title RTF Memo}{\\author Jane}}Hello RTF world.}", "application/rtf")},
    )
    assert rtf_up.status_code == 201
    rtf_da = client.post(f"/api/analysis/document/{rtf_up.json()['case_id']}").json()["document_analysis"]
    assert rtf_da["metadata"]["title"] == "RTF Memo"
    assert rtf_da["metadata"]["author"] == "Jane"

    # 5. ODT upload & analysis
    odt_up = client.post(
        "/api/media/upload",
        files={"file": ("brief.odt", create_synthetic_odt_bytes(), "application/vnd.oasis.opendocument.text")},
    )
    assert odt_up.status_code == 201
    odt_da = client.post(f"/api/analysis/document/{odt_up.json()['case_id']}").json()["document_analysis"]
    assert odt_da["metadata"]["title"] == "OpenDocument Brief"
    assert odt_da["embedded_content"]["images"]["count"] == 1

    # 6. Legacy DOC behavior (returns status='not_available' cleanly)
    doc_up = client.post(
        "/api/media/upload",
        files={"file": ("legacy.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1LEGACY_DOC", "application/msword")},
    )
    assert doc_up.status_code == 201
    doc_da = client.post(f"/api/analysis/document/{doc_up.json()['case_id']}").json()["document_analysis"]
    assert doc_da["status"] == "not_available"
    assert doc_da["parser_status"]["status"] == "not_available"

    # 7. Cross-modality rejections (image, video, audio sent to document endpoint -> 400)
    img_buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(img_buf, format="PNG")
    img_id = client.post("/api/media/upload", files={"file": ("p.png", img_buf.getvalue(), "image/png")}).json()["case_id"]
    assert client.post(f"/api/analysis/document/{img_id}").status_code == 400

    vid_id = client.post("/api/media/upload", files={"file": ("v.mp4", b"\x00\x00\x00\x14ftypisom", "video/mp4")}).json()["case_id"]
    assert client.post(f"/api/analysis/document/{vid_id}").status_code == 400

    aud_id = client.post("/api/media/upload", files={"file": ("a.mp3", b"ID3\x04\x00\x00", "audio/mpeg")}).json()["case_id"]
    assert client.post(f"/api/analysis/document/{aud_id}").status_code == 400


def test_document_analysis_error_handling_corrupt_sha256_and_missing():
    """
    Tests:
    - nonexistent case (404)
    - corrupt PDF (422)
    - corrupt DOCX (422)
    - SHA-256 mismatch before analysis (409)
    - missing file on disk (404)
    """
    assert client.post("/api/analysis/document/MT-20990101-DEADBEEF").status_code == 404

    corrupt_pdf_id = client.post(
        "/api/media/upload",
        files={"file": ("bad.pdf", b"NOT_A_PDF_HEADER", "application/pdf")},
    ).json()["case_id"]
    assert client.post(f"/api/analysis/document/{corrupt_pdf_id}").status_code == 422

    corrupt_docx_id = client.post(
        "/api/media/upload",
        files={"file": ("bad.docx", b"NOT_A_ZIP_PACKAGE", "application/octet-stream")},
    ).json()["case_id"]
    assert client.post(f"/api/analysis/document/{corrupt_docx_id}").status_code == 422

    txt_case = client.post(
        "/api/media/upload",
        files={"file": ("integrity.txt", b"Original forensic text.", "text/plain")},
    ).json()
    case_id = txt_case["case_id"]
    stored_path = Path(settings.storage_root) / "uploads" / txt_case["stored_filename"]
    orig_bytes = stored_path.read_bytes()

    try:
        stored_path.write_bytes(orig_bytes + b"TAMPERED")
        assert client.post(f"/api/analysis/document/{case_id}").status_code == 409
    finally:
        stored_path.write_bytes(orig_bytes)

    stored_path.unlink()
    try:
        assert client.post(f"/api/analysis/document/{case_id}").status_code == 404
    finally:
        stored_path.write_bytes(orig_bytes)

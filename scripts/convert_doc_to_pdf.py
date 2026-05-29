"""One-off Word .doc → .pdf converter using MS Word COM automation.

Why a separate script?
    The QCVN 41:2024 technical content arrived as `Quy chuan.doc` — a legacy
    OLE2/CFB Word document. None of the pure-Python libraries
    (`python-docx`, `mammoth`, `docx2txt`) read the binary .doc format;
    they all require .docx (zip-based). The most reliable path on a
    Windows machine with Office installed is to drive Word via COM.

    After conversion the resulting PDF is processed by the SAME pipeline
    that handles `Nghidinh_100_123_hopnhat_03.pdf` and `QCVN_41_2024.pdf`
    — no new ingestion code is required.

Usage:
    python scripts/convert_doc_to_pdf.py <input.doc> <output.pdf>
"""
from __future__ import annotations

import sys
from pathlib import Path


def doc_to_pdf(input_path: Path, output_path: Path) -> None:
    import pythoncom  # noqa: F401  — needed for COM init on some boxes
    from win32com.client import DispatchEx

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    word = DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0  # wdAlertsNone
    try:
        doc = word.Documents.Open(
            str(input_path),
            ReadOnly=True,
            AddToRecentFiles=False,
            ConfirmConversions=False,
        )
        try:
            # wdFormatPDF = 17
            doc.SaveAs2(str(output_path), FileFormat=17)
        finally:
            doc.Close(SaveChanges=0)
    finally:
        word.Quit()


def main() -> None:
    # Force UTF-8 stdout so Vietnamese paths / arrows print cleanly on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    print(f"Converting: {in_path} -> {out_path}")
    doc_to_pdf(in_path, out_path)
    print(f"Done.  Output size: {out_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

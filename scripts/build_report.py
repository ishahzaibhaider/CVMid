"""Build the VeriVision project report from its Markdown source.

Pipeline
--------
    VeriVision_Project_Report.md
        --(pandoc)-->  VeriVision_Project_Report.docx     (Word document)
        --(pandoc)-->  VeriVision_Project_Report.html      (styled, for PDF)
        --(Chrome)-->  VeriVision_Project_Report.pdf       (final PDF)

Requirements
------------
* ``pandoc`` on PATH        — Markdown -> DOCX / HTML.
* Google Chrome             — headless HTML -> PDF (no LaTeX needed).

Both the DOCX and the PDF embed every figure referenced by the report.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
STEM = "VeriVision_Project_Report"
MD = f"{STEM}.md"
DOCX = f"{STEM}.docx"
HTML = f"{STEM}.html"
PDF = f"{STEM}.pdf"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chrome") or "",
]


def _run(cmd: list[str], cwd: Path) -> None:
    print("  $", " ".join(c if " " not in c else f'"{c}"' for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def find_chrome() -> str | None:
    for path in CHROME_CANDIDATES:
        if path and Path(path).exists():
            return path
    return None


def build_docx() -> bool:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("  ! pandoc not found — skipping DOCX. Install it: brew install pandoc")
        return False
    print("[1/3] Markdown -> DOCX")
    _run([pandoc, MD, "-o", DOCX, "--toc", "--toc-depth=2",
          "--resource-path=.", "--metadata", "lang=en"], cwd=REPORTS_DIR)
    print(f"      wrote reports/{DOCX}\n")
    return True


def build_html() -> bool:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return False
    print("[2/3] Markdown -> HTML (styled)")
    _run([pandoc, MD, "-o", HTML, "--standalone", "--toc", "--toc-depth=2",
          "--css=report_style.css", "--resource-path=.",
          "--metadata", "title-prefix=VeriVision"], cwd=REPORTS_DIR)
    print(f"      wrote reports/{HTML}\n")
    return True


def build_pdf() -> bool:
    chrome = find_chrome()
    if not chrome:
        print("  ! Google Chrome not found — skipping PDF.")
        print("    Open the HTML in any browser and print to PDF instead.")
        return False
    if not (REPORTS_DIR / HTML).exists():
        print("  ! HTML not built — cannot produce PDF.")
        return False
    print("[3/3] HTML -> PDF (headless Chrome)")
    html_uri = (REPORTS_DIR / HTML).as_uri()
    _run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
          "--run-all-compositor-stages-before-draw",
          "--virtual-time-budget=20000",
          f"--print-to-pdf={PDF}", html_uri], cwd=REPORTS_DIR)
    print(f"      wrote reports/{PDF}\n")
    return True


def main() -> None:
    if not (REPORTS_DIR / MD).exists():
        sys.exit(f"Report source not found: reports/{MD}")

    print(f"Building the project report from reports/{MD}\n")
    ok_docx = build_docx()
    ok_html = build_html()
    ok_pdf = build_pdf() if ok_html else False

    print("Summary")
    for name, ok in [(DOCX, ok_docx), (PDF, ok_pdf)]:
        path = REPORTS_DIR / name
        if ok and path.exists():
            print(f"  OK   reports/{name}  ({path.stat().st_size // 1024} KB)")
        else:
            print(f"  --   reports/{name}  (not produced)")


if __name__ == "__main__":
    main()

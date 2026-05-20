# Building / exporting the project report

The full project report is **`VeriVision_Project_Report.md`** (Markdown
source). It builds to both a Word document and a PDF, each with all 41 figures
embedded.

## One command (recommended)

From the project root:

```bash
python -m scripts.build_report
```

This produces:

- `reports/VeriVision_Project_Report.docx` — the Word document
- `reports/VeriVision_Project_Report.pdf` — the PDF
- `reports/VeriVision_Project_Report.html` — intermediate, used for the PDF

It needs the **`pandoc`** CLI (`brew install pandoc`) and **Google Chrome**
(used headlessly to render the PDF — no LaTeX required).

## Manual PDF export (fallback)

If the script cannot find Chrome, open the generated HTML in any browser and
print to PDF:

1. Open `reports/VeriVision_Project_Report.html`.
2. Press **⌘ + P** → Destination **Save as PDF** → paper size **A4**.
3. Enable **Background graphics** so the cover and headings keep their colour.
4. Save as `VeriVision_Project_Report.pdf`.

## Quick check before submitting

- Cover page shows all three team names and registration numbers.
- The table of contents lists Sections 1–15.
- Every figure renders (no broken-image icons) — 41 in total.
- Each numbered section starts on a fresh page.

To regenerate the figures the report embeds:

```bash
python -m scripts.run_classical --unit all     # Units 1-7 (classical CV)
python -m scripts.verivision_figures           # VeriVision data figures
```

*Note: `VeriVision_Report.html` is the earlier mid-term report covering only
the deep-learning detector. It is superseded by
`VeriVision_Project_Report.*`, which covers the whole syllabus.*

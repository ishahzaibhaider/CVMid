# How to turn the report into a PDF

The report lives at [`reports/VeriVision_Report.html`](VeriVision_Report.html). It already embeds every figure from `reports/figures/` and is styled like an academic paper. Converting to PDF takes ~30 seconds with zero installs.

## Easiest method — browser print-to-PDF (recommended)

1. In Finder, navigate to `~/CVMid/reports/` and **double-click `VeriVision_Report.html`**. It opens in Safari or Chrome.
2. Press **⌘ + P** (Cmd + P).
3. In the print dialog:
   - **Destination / PDF:** "Save as PDF"
   - **Paper size:** US Letter
   - **Margins:** Default (the page CSS already sets correct margins)
   - **Background graphics / Print backgrounds:** **Enabled** (so the cover page colours print)
4. Click **Save** → name it `VeriVision_Report.pdf` → done.

Result: a ~10-page PDF with cover page, abstract, figures, tables, and references. This is the file you submit.

## Alternative — use Chrome's headless mode (one command)

If the print dialog is being awkward:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --headless --disable-gpu \
  --print-to-pdf=VeriVision_Report.pdf \
  file://$HOME/CVMid/reports/VeriVision_Report.html
```

Run that from any terminal — it spits out `VeriVision_Report.pdf` in the current directory.

## Quick-check before submitting

Open the generated PDF and make sure:

- Cover page shows all three team names and registrations correctly.
- Training-curve, confusion-matrix, and ROC figures all render (not broken image icons).
- Page breaks fall at section boundaries (Methodology, Results, Limitations all start on a new page).
- No page is blank or cut off mid-sentence.

If any figure is missing, it's because its PNG wasn't in `reports/figures/` when you opened the HTML. Verify with:

```bash
ls reports/figures/
```

Expected files: `confusion_custom_cnn.png`, `confusion_resnet50.png`, `roc_custom_cnn.png`, `roc_resnet50.png`, `training_curves_custom_cnn.png`, `training_curves_resnet50.png`.

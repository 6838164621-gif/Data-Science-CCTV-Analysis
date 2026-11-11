# Automatic CCTV cleaning

This is the default workflow when you want a complete first pass without manually reviewing images. It uses configurable quality rules. The earlier experimental learned classifier is not used because it performed poorly. These rules are not a validated object-countability model; tune them as needed later.

## Run in VS Code on your Mac

Open this folder in VS Code. In the terminal:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_auto.txt
```

If you do not already have a scan manifest, create one:

```sh
python clean_cctv.py scan \
  --source "/Users/irinrayasotangkur/Desktop/DataSciCCTV/cctv_images.zip" \
  --output scan_results --review-limit 0
```

Then run automatic cleaning:

```sh
python automate_cleaner.py \
  --manifest scan_results/manifest.json \
  --decisions review_decisions.json \
  --output automatic_results --apply
```

The package includes your downloaded `review_decisions.json`. It supplements the 12 earlier reference decisions in the scan. Omit `--decisions review_decisions.json` for a different dataset where these decisions do not apply. Select fresh output folders for each run.

If your existing scan manifest is elsewhere, use its absolute path instead of `scan_results/manifest.json`.

## What happens

Every image receives a keep, adjust, or drop decision. There is no mandatory review step. Your explicit saved decisions take precedence over automatic rules.

- Drop unreadable/blank frames, broadcast color test patterns, substantial gray corruption, near-featureless images, and scenes where most sampled areas remain unreadable after a contrast trial.
- Noise alone never triggers deletion. The six previously rejected noisy frames are explicitly recorded as keep decisions in the included decisions file.
- Retain usable night and rain images, moderate blur, localized defects, small objects, and ordinary occlusion. Ambiguous quality is retained for later fine-tuning.
- Apply the same luminance-only CLAHE adjustment used in the accepted examples selectively to low-contrast/dark images. Avoid using it on images with high estimated noise. No generative enhancement or aggressive denoising is used.
- For darkness, sample 48 tiles across the frame below the top 12% and above the bottom 5%, including both side areas. Dark tiles must retain brightness, local contrast, and coarse edge structure after a CLAHE trial; if more than 55% of sampled tiles remain unreadable, drop the frame. Useful recovery can select the adjusted version. These are scene-area proxies, not semantic road or sidewalk masks. Local lights only affect their own tiles.
- Preserve the source ZIP/folder. Create a working dataset; rejected images are deleted from that copy and adjusted frames are stored as lossless PNGs.

The output contains:

- `automated_manifest.json`: complete decisions, reasons, and extra quality measurements.
- `automation_summary.json`: totals, reasons, and breakdowns by camera and filename hour.
- `cleaned_dataset/images/`: the actual remaining images.
- `cleaned_dataset/audit.json`: each deletion, retention, and adjustment with its reason and output path.
- `cleaned_dataset/summary.json`: applied totals.

The ZIP is not changed. Check `complete: true` in the audit before using the output. Source files are checked against the original scan; source changes stop the run. If interrupted, rerun into a fresh output directory.

## Fine-tune later

Edit `quality_rules.json` and rerun into a fresh output directory. The gray-fill threshold and contrast limit control rejection of large neutral gray corruption areas. Noise measurements remain diagnostics only. Darkness requires a large unreadable area after adjustment; nighttime alone is not a rejection reason. General measurements use 160 × 120 samples; darkness uses 48 tiles from a 240 × 180 scene crop and checks spatial structure after smoothing to reduce sensitivity to grain. These measurements cannot establish whether every pedestrian or vehicle is countable. Camera/sidewalk regions are not defined yet. No accuracy percentage is claimed for this rule model.

Brightness/contrast adjustments cannot recover missing detail, and local contrast enhancement can amplify noise; this is why this version avoids it when the noise proxy is high. [OpenCV explanation of CLAHE](https://docs.opencv.org/4.10.0/d2/d74/tutorial_js_histogram_equalization.html).

Density counting and low/medium/high classification remain separate later steps.

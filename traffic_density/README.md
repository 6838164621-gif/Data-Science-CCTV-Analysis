# CCTV traffic density

## Run locally

Use Python 3.12. Open Terminal in this folder and run:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt -o yolo26m.pt
python -m streamlit run app.py
```

If the sidebar already finds the local YOLO26m checkpoint downloaded during testing, you can skip the download. If you have another local copy of the same weights, enter its path in the sidebar. The model is [Ultralytics YOLO26m](https://docs.ultralytics.com/models/yolo26/).

1. Set **Cleaned image folder** to `~/Desktop/Cleaned_CCTV/cctv_cleaner/automatic_results/cleaned_dataset/images` or `~/Desktop/Cleaned_CCTV/cleaned_cctv_images`. Both local copies contain 60,737 images in 100 camera folders. For a new installation, extract the ZIP and point the app at the extracted folder. The app accepts either camera folders directly under the selected folder or an intermediate folder such as `cctv_images`.
2. Set **LabelMe drafts folder** to your `Data/Labelme workflow/drafts` directory. Match each camera ID to one JSON file, such as `135.json`. Use LabelMe polygon labels `road`, `crosswalk`, `highway`, and `sidewalk` as applicable. Currently 17 camera angles have sidewalk polygons. Camera 1028 is one of them. The dashboard lists missing, invalid, or duplicate camera annotations and withholds density for those cameras until the issue is fixed.
3. Optionally set **Camera metadata CSV** to your `cctv_info.csv` for location names. The catalog uses the actual folders and images on disk.
4. Choose a **Camera angle** by camera ID or location, then set the start and end dates and times. Select **Process this camera angle**. The first run processes all cleaned frames from that angle to establish its vehicle and sidewalk pedestrian maxima. Leave the browser tab open during inference. If interrupted, select the same angle and run it again with the same detection settings, paths, and SQLite database to reuse saved counts.
5. The hourly bar charts show road, highway, and sidewalk densities for the selected dates and times. Each bar averages the frame densities within one local calendar hour. Highest and lowest hours in the selected period are highlighted separately in each chart, including ties. When all hours have the same value, they use the same color. The highway chart appears for camera angles with a LabelMe highway polygon. The table and frame inspector show frames in the selected period. Changing the dates or times uses the existing results immediately. For a teammate handoff, download the full-angle CSV so it contains every frame for that camera. Use the selected-period CSV for a filtered export. If the laptop has an NVIDIA GPU and a CUDA-enabled PyTorch installation, select `cuda:0` under **Detection settings → Device**. CSV values are percentages on a 0–100 scale.

## How counting works

The detector uses the YOLO26m pipeline from `app 3.py`, including multi-pass inference, rider pairing, motorcycle recovery, and duplicate merging. The car group includes buses and trucks. COCO bicycle boxes are used only to suppress cyclist false positives. A rider with no recognized bicycle can still trigger the original rider-inference heuristic. Disable **Infer motorcycles from rider evidence** if that heuristic produces false positives in your sample.

The polygon mask and bottom-edge fallback come from `app_uvh26.py`. Later polygons in the JSON take precedence where shapes overlap. A vehicle belongs to one region. Region membership uses its bottom-center point, using the motorcycle box before it is expanded to display the rider. The bottom-edge fallback samples nearby labeled pixels under the original rule. Use polygons drawn for the same view and crop as the cleaned images.

The detector reuses YOLO26m person detections to count pedestrians whose feet fall inside the sidewalk polygons. People paired with detected or recovered motorcycles and recognized cyclists are excluded. Angles without a sidewalk polygon show sidewalk density as unavailable and have blank sidewalk values in the CSV.

For each camera, the app scans all cleaned frames and stores separate maxima for road + crosswalk vehicles, highway vehicles, and sidewalk pedestrians. Each density is 100 times the matching per-image count divided by that maximum, or 0% if the maximum is zero. The CSV includes the denominators and the vehicle breakdown by region. Dashboard filters act on rows after density is calculated. A damaged frame prevents its camera from publishing results until the scan can complete.

The local database stores per-frame counts, detections, and camera maxima. **Refresh inventory** checks for added, changed, or removed files. **Process this camera angle** refreshes automatically. The app reuses compatible checkpoints from this version and recomputes maxima for the selected angle from its current inventory. Runs made before sidewalk tracking need to be processed again because the old checkpoints lack pedestrian detections.

## Validate a small sample

Copy a few cleaned images into a separate folder while keeping their camera folders. Point the app at that sample and the corresponding camera JSONs, then run an angle and inspect the overlays and CSV. The sample has its own baseline because the app keys baselines by image root. For production, switch back to the full cleaned-image path and process the angle you want to analyze.

from pathlib import Path
import json
import re

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
# Use the cleaned working dataset produced by cctv_cleaner. The original
# dataset remains untouched and can be restored by changing this path.
IMAGE_ROOT = ROOT / "cctv_cleaner" / "automatic_results" / "cleaned_dataset" / "images"
MODEL_DIR = ROOT / "models" / "uvh26"
MODEL_PATH = MODEL_DIR / "UVH-26-MV-YOLOv11-S.pt"
PERSON_MODEL_PATH = ROOT / "yolo11x.pt"
SEGFORMER_DIR = ROOT / "segformer_no_sidewalk_training" / "model"
LABELME_DIR = ROOT / "Data" / "Labelme workflow" / "drafts"
CLASS_FILE = MODEL_DIR / "uvh_classes.txt"

GROUPS = {
    "Hatchback": "Car", "Sedan": "Car", "SUV": "Car", "MUV": "Car", "Three-wheeler" : "Car",
    "Truck": "Truck", "LCV": "Truck",
    "Bus": "Bus", "Mini-bus": "Bus", "tempo-traveller": "Bus",
    "Van": "Van", "Two-wheeler": "Motorcycle",
    "Bicycle": "Bicycle", "Others": "Other", "person": "Pedestrian",
}
GROUP_COLORS = {
    "Car": (40, 180, 255), "Truck": (255, 170, 45), "Bus": (100, 220, 100),
    "Van": (235, 100, 220), "Motorcycle": (255, 90, 90),
    "Three-wheeler": (180, 150, 255), "Bicycle": (80, 220, 210), "Other": (190, 190, 190),
    "Pedestrian": (255, 240, 70),
}

st.set_page_config(page_title="UVH-26 CCTV detector", layout="wide")
st.title("UVH-26 CCTV object detection")
st.caption("Choose a CCTV frame to inspect vehicle detections. Region masks are not applied yet.")


@st.cache_resource(show_spinner=False)
def load_detector(model_path):
    from ultralytics import YOLO
    return YOLO(str(model_path))


@st.cache_resource(show_spinner=False)
def load_sahi_detector(model_path):
    from sahi import AutoDetectionModel
    import torch
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=0.05,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )


@st.cache_resource(show_spinner=False)
def load_segformer():
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch
    processor = SegformerImageProcessor.from_pretrained(str(SEGFORMER_DIR))
    model = SegformerForSemanticSegmentation.from_pretrained(str(SEGFORMER_DIR))
    model.eval()
    return processor, model, torch


def load_labelme_regions(camera_id, image_size):
    """Rasterize the manually edited LabelMe polygons for one camera."""
    json_path = LABELME_DIR / f"{camera_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No LabelMe annotation found for camera {camera_id}: {json_path}")
    annotation = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(annotation.get("imageWidth", image_size[0]))
    height = int(annotation.get("imageHeight", image_size[1]))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    label_ids = {"road": 1, "highway": 2, "crosswalk": 3}
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        points = shape.get("points", [])
        if label not in label_ids or len(points) < 3:
            continue
        draw.polygon([(float(x), float(y)) for x, y in points], fill=label_ids[label])
    return __import__("numpy").array(mask.resize(image_size, Image.Resampling.NEAREST))


def region_overlay(image, region_mask):
    """Show SegFormer classes transparently while preserving the CCTV image."""
    colors = {
        0: (0, 0, 0, 0),
        1: (40, 130, 255, 75),   # road
        2: (255, 150, 30, 90),   # highway
        3: (220, 50, 180, 105),  # crosswalk
    }
    rgba = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pixels = rgba.load()
    for y in range(region_mask.shape[0]):
        for x in range(region_mask.shape[1]):
            pixels[x, y] = colors.get(int(region_mask[y, x]), (0, 0, 0, 0))
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")


def point_region(detection, region_mask):
    x1, y1, x2, y2 = detection["box"]
    x = max(0, min(region_mask.shape[1] - 1, int((x1 + x2) / 2)))
    y = max(0, min(region_mask.shape[0] - 1, int(y2)))
    return int(region_mask[y, x])


def overlaps_detection(first, second, threshold=0.15):
    return box_iou(first["box"], second["box"]) >= threshold


@st.cache_data
def image_catalog():
    rows = []
    for path in IMAGE_ROOT.glob("*/*.jpg"):
        match = re.search(r"_(\d{8})_(\d{6})", path.name)
        if not match:
            continue
        rows.append({
            "camera_id": path.parent.name,
            "date": pd.to_datetime(match.group(1), format="%Y%m%d").date(),
            "time": pd.to_datetime(match.group(2), format="%H%M%S").time(),
            "path": str(path),
            "label": f"Camera {path.parent.name} — {match.group(1)} {match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:]}",
        })
    return pd.DataFrame(rows).sort_values(["camera_id", "date", "time"]).reset_index(drop=True)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def prepare_detections(result, image_size, nested_overlap, allowed_classes=None):
    """Clip boxes and remove same-category duplicates/nested boxes."""
    width, height = image_size
    candidates = []
    if result.boxes is None:
        return candidates
    for box, cls, conf in zip(
        result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()
    ):
        raw = result.names[int(cls)]
        if allowed_classes is not None and raw not in allowed_classes:
            continue
        clipped = [
            max(0.0, min(float(box[0]), width - 1)),
            max(0.0, min(float(box[1]), height - 1)),
            max(0.0, min(float(box[2]), width - 1)),
            max(0.0, min(float(box[3]), height - 1)),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        candidates.append({"box": clipped, "raw_class": raw, "group": GROUPS.get(raw, "Other"), "confidence": float(conf)})

    kept = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        duplicate = False
        for existing in kept:
            if candidate["group"] != existing["group"]:
                continue
            overlap = box_iou(candidate["box"], existing["box"])
            x1, y1, x2, y2 = candidate["box"]
            ex1, ey1, ex2, ey2 = existing["box"]
            intersection = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
            candidate_area = max(1, (x2 - x1) * (y2 - y1))
            contained = intersection / candidate_area
            if overlap >= nested_overlap or contained >= 0.80:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def prepare_sahi_detections(sliced_result, image_size, nested_overlap, allowed_classes=None):
    """Convert SAHI predictions into the same format as regular inference."""
    class_names = {}
    candidates = []
    for prediction in sliced_result.object_prediction_list:
        box = prediction.bbox.to_xyxy()
        raw = prediction.category.name
        if allowed_classes is not None and raw not in allowed_classes:
            continue
        candidates.append({
            "box": [float(value) for value in box],
            "raw_class": raw,
            "group": GROUPS.get(raw, "Other"),
            "confidence": float(prediction.score.value),
        })

    # Reuse the same clipping and duplicate suppression logic by applying it
    # to a small adapter object with the format expected by prepare_detections.
    class BoxValues:
        def __init__(self, values):
            self._values = values

        def cpu(self):
            return self

        def tolist(self):
            return self._values

    class ResultAdapter:
        names = {index: name for index, name in enumerate(class_names.values())}
        boxes = None

    # SAHI has already merged tile predictions, so perform clipping and the
    # same-category nesting check directly.
    width, height = image_size
    clipped = []
    for item in candidates:
        item["box"] = [
            max(0.0, min(item["box"][0], width - 1)), max(0.0, min(item["box"][1], height - 1)),
            max(0.0, min(item["box"][2], width - 1)), max(0.0, min(item["box"][3], height - 1)),
        ]
        if item["box"][2] > item["box"][0] and item["box"][3] > item["box"][1]:
            clipped.append(item)
    kept = []
    for candidate in sorted(clipped, key=lambda item: item["confidence"], reverse=True):
        duplicate = False
        for existing in kept:
            if candidate["group"] != existing["group"]:
                continue
            overlap = box_iou(candidate["box"], existing["box"])
            x1, y1, x2, y2 = candidate["box"]
            ex1, ey1, ex2, ey2 = existing["box"]
            intersection = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
            area = max(1, (x2 - x1) * (y2 - y1))
            if overlap >= nested_overlap or intersection / area >= 0.80:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def transparent_overlay(image, detections, show_grouped):
    """Draw thin, translucent detections so the CCTV scene stays visible."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default(size=10)
    except TypeError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        color = GROUP_COLORS[detection["group"]]
        # Thin, partially transparent outline.
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 175), width=2)
        label_name = detection["group"] if show_grouped else detection["raw_class"]
        if detection.get("analysis", "").startswith("excluded"):
            label_name = f"{label_name} [excluded]"
        label = f"{label_name} {detection['confidence']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        label_y = max(0, y1 - text_height - 4)
        draw.rounded_rectangle(
            (x1, label_y, x1 + text_width + 6, label_y + text_height + 4),
            radius=2,
            fill=(*color, 155),
        )
        draw.text((x1 + 3, label_y + 2), label, fill=(255, 255, 255, 235), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


if not IMAGE_ROOT.exists():
    st.error(f"Could not find CCTV images at: {IMAGE_ROOT}")
    st.stop()

catalog = image_catalog()
if catalog.empty:
    st.error("No timestamped JPG images were found.")
    st.stop()

with st.sidebar:
    st.header("Image selection")
    selected_models = st.multiselect(
        "Models to display",
        ["UVH-26 vehicles", "YOLO11x pedestrians"],
        default=["UVH-26 vehicles", "YOLO11x pedestrians"],
    )
    cameras = sorted(catalog.camera_id.unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    camera = st.selectbox("Camera ID", cameras)
    camera_rows = catalog[catalog.camera_id == camera]
    dates = sorted(camera_rows.date.unique())
    selected_date = st.selectbox("Date", dates)
    date_rows = camera_rows[camera_rows.date == selected_date]
    labels = date_rows.label.tolist()
    selected_label = st.selectbox("Image time", labels)
    confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("NMS IoU", 0.10, 0.95, 0.45, 0.05)
    nested_overlap = st.slider("Duplicate suppression overlap", 0.30, 0.95, 0.55, 0.05)
    show_grouped = st.toggle("Show broad vehicle categories", value=True)
    region_analysis = st.toggle("Use region-aware combined analysis", value=False)
    show_regions = st.toggle("Show saved LabelMe road markings", value=True)
    use_sahi = True
    slice_size = 320
    slice_overlap = 0.25
    st.caption("SAHI: slice size 320 · overlap 0.25")

if not selected_models:
    st.warning("Select at least one model from the sidebar.")
    st.stop()

model_choice = selected_models[0]
show_uvh = "UVH-26 vehicles" in selected_models
show_yolo = "YOLO11x pedestrians" in selected_models
run_combined = region_analysis or (show_uvh and show_yolo)

selected = date_rows[date_rows.label == selected_label].iloc[0]
image_path = Path(selected.path)
active_model_path = MODEL_PATH if model_choice == "UVH-26 vehicles" else PERSON_MODEL_PATH

st.write(f"**Selected:** camera `{camera}` · `{selected.date}` · `{selected.time}`")
image = Image.open(image_path).convert("RGB")
region_mask = None
if show_regions and not run_combined:
    try:
        with st.spinner("Loading saved LabelMe regions…"):
            region_mask = load_labelme_regions(camera, image.size)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
with st.spinner(f"Running {'combined region analysis' if region_analysis else model_choice}…"):
    display_detections = None
    if run_combined:
        try:
            region_mask = load_labelme_regions(camera, image.size)
        except FileNotFoundError as error:
            st.error(str(error))
            st.stop()
        if use_sahi:
            from sahi.predict import get_sliced_prediction
            vehicle_sahi = load_sahi_detector(str(MODEL_PATH))
            person_sahi = load_sahi_detector(str(PERSON_MODEL_PATH))
            vehicle_sahi.confidence_threshold = confidence
            person_sahi.confidence_threshold = confidence
            vehicle_result = get_sliced_prediction(image, vehicle_sahi, slice_height=slice_size, slice_width=slice_size, overlap_height_ratio=slice_overlap, overlap_width_ratio=slice_overlap, postprocess_type="GREEDYNMM", postprocess_match_metric="IOS", postprocess_match_threshold=iou, verbose=0)
            person_result = get_sliced_prediction(image, person_sahi, slice_height=slice_size, slice_width=slice_size, overlap_height_ratio=slice_overlap, overlap_width_ratio=slice_overlap, postprocess_type="GREEDYNMM", postprocess_match_metric="IOS", postprocess_match_threshold=iou, verbose=0)
            vehicle_detections = prepare_sahi_detections(vehicle_result, image.size, nested_overlap) if show_uvh else []
            person_detections = prepare_sahi_detections(person_result, image.size, nested_overlap, {"person"}) if show_yolo else []
        else:
            vehicle_result = load_detector(str(MODEL_PATH)).predict(source=image, conf=confidence, iou=iou, verbose=False)[0]
            person_result = load_detector(str(PERSON_MODEL_PATH)).predict(source=image, conf=confidence, iou=iou, classes=[0], verbose=False)[0]
            vehicle_detections = prepare_detections(vehicle_result, image.size, nested_overlap) if show_uvh else []
            person_detections = prepare_detections(person_result, image.size, nested_overlap, {"person"}) if show_yolo else []

        motorcycles = [item for item in vehicle_detections if item["raw_class"] == "Two-wheeler"]
        detections = []
        for item in vehicle_detections:
            if point_region(item, region_mask) == 0 and item["raw_class"] == "Two-wheeler":
                item["analysis"] = "motorcycle in background"
                detections.append(item)
            else:
                item["analysis"] = "excluded: not a background motorcycle"
        for item in person_detections:
            region = point_region(item, region_mask)
            if region in {1, 2, 3} and not any(overlaps_detection(item, motorcycle) for motorcycle in motorcycles):
                item["analysis"] = "pedestrian in road region, not in motorcycle"
                detections.append(item)
            elif region == 0:
                item["analysis"] = "pedestrian in background region"
                detections.append(item)
            elif region in {1, 2, 3}:
                item["analysis"] = "excluded: overlaps motorcycle in road region"
            else:
                item["analysis"] = "excluded: region not selected"
        # Show raw detections from both models, while `detections` remains the
        # filtered set used for the analysis count.
        display_detections = vehicle_detections + person_detections
    elif use_sahi:
        from sahi.predict import get_sliced_prediction
        sahi_detector = load_sahi_detector(str(active_model_path) if active_model_path.exists() else "yolo11x.pt")
        sahi_detector.confidence_threshold = confidence
        sliced = get_sliced_prediction(
            image,
            sahi_detector,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=slice_overlap,
            overlap_width_ratio=slice_overlap,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=iou,
            verbose=0,
        )
        detections = prepare_sahi_detections(
            sliced, image.size, nested_overlap,
            allowed_classes={"person"} if model_choice == "YOLO11x pedestrians" else None,
        )
    else:
        detector = load_detector(str(active_model_path) if active_model_path.exists() else "yolo11x.pt")
        result = detector.predict(
            source=image,
            conf=confidence,
            iou=iou,
            classes=[0] if model_choice == "YOLO11x pedestrians" else None,
            verbose=False,
        )[0]
        detections = prepare_detections(
            result, image.size, nested_overlap,
            allowed_classes={"person"} if model_choice == "YOLO11x pedestrians" else None,
        )
        display_detections = detections

display_image = region_overlay(image, region_mask) if show_regions and region_mask is not None else image
annotated = transparent_overlay(display_image, display_detections or detections, show_grouped)
left, right = st.columns([1.6, 1])
with left:
    st.image(annotated, caption="Detections", use_container_width=True)
    if show_regions and region_mask is not None:
        st.caption("Saved LabelMe regions: 🟦 road · 🟧 highway · 🟪 crosswalk · unshaded background")
with right:
    records = []
    included_ids = {id(item) for item in detections}
    for detection in (display_detections or detections):
        box = detection["box"]
        records.append({
            "class": detection["group"] if show_grouped else detection["raw_class"],
            "original_class": detection["raw_class"],
            "analysis": detection.get("analysis", "unfiltered model detection"),
            "included_for_analysis": id(detection) in included_ids,
            "confidence": round(detection["confidence"], 3),
            "x1": round(box[0], 1), "y1": round(box[1], 1),
            "x2": round(box[2], 1), "y2": round(box[3], 1),
        })
    st.metric("Included for analysis", len(detections))
    if run_combined:
        st.caption(f"Displaying all model detections: {len(display_detections or [])}. Excluded detections remain visible for verification.")
    if records:
        table = pd.DataFrame(records).sort_values("confidence", ascending=False)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("Download detections CSV", table.to_csv(index=False), "uvh26_detections.csv", "text/csv")
    else:
        st.info("No objects passed the confidence threshold.")

with st.expander("About this model"):
    if model_choice == "UVH-26 vehicles":
        st.write("UVH-26 detects vehicle categories such as hatchback, sedan, SUV, bus, truck, two-wheeler, bicycle, van, and others. It is being used here as a general detector over the whole image; zone-specific analysis will be added later.")
    else:
        st.write("YOLO11x is used with its COCO-trained person class. Pedestrian detections are not yet assigned to SegFormer regions.")

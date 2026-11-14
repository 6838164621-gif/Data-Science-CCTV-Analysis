"""LabelMe rasterization and region membership adapted from app_uvh26.py."""
import json
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

LABEL_IDS = {"road": 1, "highway": 2, "crosswalk": 3}
REGION_NAMES = {0: "background", **{i: name for name, i in LABEL_IDS.items()}}
SIDEWALK_ID = 4
ANNOTATION_LABELS = {**LABEL_IDS, "sidewalk": SIDEWALK_ID}
BOTTOM_UNKNOWN_RATIO = 0.05
COLORS = [(0, 0, 0, 0), (40, 130, 255, 75), (255, 150, 30, 90),
          (220, 50, 180, 105), (55, 225, 149, 100)]


def read_annotation(path):
    annotation = json.loads(path.read_text(encoding="utf-8-sig"))
    shapes = annotation["shapes"]
    usable = False
    for shape in shapes:
        if str(shape.get("label", "")).strip().lower() in ANNOTATION_LABELS:
            points = shape.get("points", [])
            if len(points) >= 3:
                coords = np.asarray(points, dtype=float)
                if coords.shape != (len(points), 2) or not np.isfinite(coords).all():
                    raise ValueError("Invalid LabelMe polygon coordinates")
                usable = True
    if not usable:
        raise ValueError("Expected a road, highway, crosswalk or sidewalk polygon")
    for key in ("imageWidth", "imageHeight"):
        if key in annotation and int(annotation[key]) <= 0:
            raise ValueError(f"Invalid {key}")
    return annotation


@lru_cache(maxsize=16)
def load_labelme_regions(json_path, signature, image_size, sidewalk=False):
    """Rasterize polygons in file order. Later shapes take precedence."""
    annotation = read_annotation(json_path)
    width = int(annotation.get("imageWidth", image_size[0]))
    height = int(annotation.get("imageHeight", image_size[1]))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    labels = {"sidewalk": SIDEWALK_ID} if sidewalk else LABEL_IDS
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        points = shape.get("points", [])
        if label not in labels or len(points) < 3:
            continue
        draw.polygon([(float(x), float(y)) for x, y in points], fill=labels[label])
    return np.array(mask.resize(image_size, Image.Resampling.NEAREST))


def point_region(detection, region_mask):
    x1, y1, x2, y2 = detection.get("region_box", detection["box"])
    x = max(0, min(region_mask.shape[1] - 1, int((x1 + x2) / 2)))
    y = max(0, min(region_mask.shape[0] - 1, int(y2)))
    # LabelMe polygons may stop just above the bottom screen edge. Look upward
    # in a narrow vertical band before falling back to the sampled mask value.
    if y >= int(region_mask.shape[0] * (1 - BOTTOM_UNKNOWN_RATIO)):
        band_width = max(1, int(region_mask.shape[1] * 0.02))
        y_start = max(0, y - int(region_mask.shape[0] * 0.20))
        nearby = region_mask[y_start:y + 1, max(0, x - band_width):min(region_mask.shape[1], x + band_width + 1)]
        labeled = nearby[nearby > 0]
        if labeled.size:
            values, counts = np.unique(labeled, return_counts=True)
            return int(values[counts.argmax()])
    return int(region_mask[y, x])


def on_sidewalk(detection, mask):
    """Test where the pedestrian's feet meet the explicitly annotated sidewalk."""
    x1, _, x2, y2 = detection["box"]
    height, width = mask.shape
    x = max(0, min(width - 1, int((x1 + x2) / 2)))
    y = max(0, min(height - 1, int(y2)))
    return mask[y, x] == SIDEWALK_ID


def region_overlay(image, mask):
    rgba = Image.fromarray(np.asarray(COLORS, dtype=np.uint8)[mask])
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")

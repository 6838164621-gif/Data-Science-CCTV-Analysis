"""Discover cleaned frames, checkpoint detections and publish complete camera baselines."""
import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from PIL import Image

from regions import REGION_NAMES, load_labelme_regions, on_sidewalk, point_region, read_annotation

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
TIMESTAMP = re.compile(r"_(\d{8})_(\d{6})(?!\d)")
REGIONS = ("road", "crosswalk", "highway", "background")
CLASSES = ("car", "motorcycle")
BREAKDOWN = tuple(f"{r}_{c}" for r in REGIONS for c in CLASSES)


def fingerprint(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp(name):
    match = TIMESTAMP.search(name)
    if match:
        try:
            return datetime.strptime("_".join(match.groups()), "%Y%m%d_%H%M%S").isoformat(" ")
        except ValueError:
            pass
    return ""


def discover(root):
    cameras = {}

    def raise_error(error):
        raise error

    for directory, folders, names in os.walk(root, onerror=raise_error):
        folders[:] = [f for f in folders if not f.startswith(".") and f != "__MACOSX"]
        for name in names:
            path = Path(directory) / name
            if name.startswith(".") or path.suffix.lower() not in EXTENSIONS:
                continue
            parts = path.relative_to(root).parts
            # Both camera_id/... and a single or nested wrapper around camera folders.
            camera = next((part for part in parts[:-1] if part.isdecimal()), "")
            if not camera:
                camera = parts[0] if len(parts) > 1 else name.split("_", 1)[0]
            relative = path.relative_to(root).as_posix()
            cameras.setdefault(camera, {})[relative] = fingerprint(path)
    return {cam: dict(sorted(paths.items())) for cam, paths in sorted(cameras.items())}


def annotations(root, cameras):
    found = {}
    for path in sorted(root.rglob("*.json")):
        if path.stem in cameras:
            found.setdefault(path.stem, []).append(path)
    usable, issues = {}, {}
    for camera in cameras:
        paths = found.get(camera, [])
        if len(paths) != 1:
            issues[camera] = "Missing camera JSON" if not paths else "Multiple matching camera JSON files"
            continue
        try:
            read_annotation(paths[0])
            usable[camera] = (paths[0], file_digest(paths[0]), fingerprint(paths[0]))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            issues[camera] = str(error)
    return usable, issues


def scan_signature(settings, annotation_digest):
    return digest([settings, annotation_digest])


@contextmanager
def cache(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    try:
        with db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS frames (
                    root TEXT, camera TEXT, config TEXT, path TEXT,
                    size INTEGER, mtime INTEGER, details TEXT NOT NULL,
                    PRIMARY KEY (root, camera, config, path)
                );
                CREATE TABLE IF NOT EXISTS baselines (
                    root TEXT, camera TEXT, config TEXT, manifest TEXT,
                    road_max INTEGER, highway_max INTEGER,
                    PRIMARY KEY (root, camera, config)
                );
                CREATE TABLE IF NOT EXISTS sidewalk_baselines (
                    root TEXT, camera TEXT, config TEXT, maximum INTEGER,
                    PRIMARY KEY (root, camera, config)
                );
            """)
        yield db
    finally:
        db.close()


def frame_counts(detections, has_sidewalk):
    counts = dict.fromkeys(BREAKDOWN, 0)
    for item in detections:
        if item["name"] == "person":
            continue
        counts[f'{item["region"]}_{item["name"]}'] += 1
    counts["road_count"] = sum(counts[f"{r}_{c}"] for r in ("road", "crosswalk") for c in CLASSES)
    counts["highway_count"] = sum(counts[f"highway_{c}"] for c in CLASSES)
    counts["sidewalk_pedestrian_count"] = (
        sum(item["name"] == "person" for item in detections) if has_sidewalk else None
    )
    return counts


def inspect_frame(path, annotation, detector):
    with Image.open(path) as image:
        image = image.convert("RGB")
    json_path, _, json_signature = annotation
    mask = load_labelme_regions(json_path, json_signature, image.size)
    sidewalk = load_labelme_regions(json_path, json_signature, image.size, sidewalk=True)
    has_sidewalk = bool(sidewalk.any())
    dets = []
    for detection in detector(image):
        if detection["name"] == "person":
            if not has_sidewalk or not on_sidewalk(detection, sidewalk):
                continue
            region = "sidewalk"
        else:
            region = REGION_NAMES[point_region(detection, mask)]
        dets.append({**detection, "region": region})
    return {"detections": dets, "counts": frame_counts(dets, has_sidewalk)}


def count_dataset(db, root, cameras, labels, settings, detector, progress):
    """Checkpoint each frame. Only publish a baseline when every camera frame is counted."""
    root_key = str(root)
    errors = {}
    for camera, paths in cameras.items():
        if camera not in labels:
            continue
        annotation = labels[camera]
        config = scan_signature(settings, annotation[1])
        args = (root_key, camera, config)
        cached = {name: (size, mtime, json.loads(details)["counts"])
                  for name, size, mtime, details in db.execute(
                      "SELECT path, size, mtime, details FROM frames "
                      "WHERE root=? AND camera=? AND config=?", args)}
        road_max = highway_max = 0
        sidewalk_max = None
        for relative, signature in paths.items():
            previous = cached.get(relative)
            path = root / relative
            if previous and previous[:2] == signature:
                counts = previous[2]
            else:
                try:
                    details = inspect_frame(path, annotation, detector)
                    if fingerprint(path) != signature:
                        raise ValueError("Image changed during processing. Run again.")
                except (OSError, ValueError) as error:
                    errors.setdefault(camera, []).append((relative, str(error)))
                    progress()
                    continue
                with db:
                    db.execute("INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (*args, relative, *signature, json.dumps(details)))
                counts = details["counts"]
            road_max = max(road_max, counts["road_count"])
            highway_max = max(highway_max, counts["highway_count"])
            pedestrians = counts["sidewalk_pedestrian_count"]
            if pedestrians is not None:
                sidewalk_max = max(sidewalk_max or 0, pedestrians)
            progress()
        if fingerprint(annotation[0]) != annotation[2]:
            errors.setdefault(camera, []).append((str(annotation[0]), "Annotation changed. Run again."))
        if camera not in errors:
            with db:
                db.execute("INSERT OR REPLACE INTO baselines VALUES (?, ?, ?, ?, ?, ?)",
                           (*args, digest(paths), road_max, highway_max))
                db.execute("INSERT OR REPLACE INTO sidewalk_baselines VALUES (?, ?, ?, ?)",
                           (*args, sidewalk_max))
    return errors


def density(count, maximum):
    return 100 * count / maximum if maximum else 0.0


def report(db, root, cameras, labels, settings):
    """Read only baselines for the current inventory. UI filters run after this function."""
    rows = []
    for camera, paths in cameras.items():
        if camera not in labels:
            continue
        args = (str(root), camera, scan_signature(settings, labels[camera][1]))
        maximums = db.execute(
            "SELECT road_max, highway_max, maximum FROM baselines "
            "JOIN sidewalk_baselines USING (root, camera, config) "
            "WHERE root=? AND camera=? AND config=? AND manifest=?", (*args, digest(paths)),
        ).fetchone()
        if maximums is None:
            continue
        road, highway, sidewalk = maximums
        camera_rows = []
        for path, size, mtime, details in db.execute(
            "SELECT path, size, mtime, details FROM frames WHERE root=? AND camera=? AND config=?", args
        ):
            if paths.get(path) != (size, mtime):
                continue
            counts = json.loads(details)["counts"]
            camera_rows.append({
                "camera_id": camera, "captured_at": timestamp(path), "image": path,
                **counts, "road_max": road, "highway_max": highway,
                "road_density_pct": density(counts["road_count"], road),
                "highway_density_pct": density(counts["highway_count"], highway),
                "sidewalk_pedestrian_max": sidewalk,
                "sidewalk_pedestrian_density_pct": (
                    density(counts["sidewalk_pedestrian_count"], sidewalk)
                    if counts["sidewalk_pedestrian_count"] is not None else None
                ),
            })
        if len(camera_rows) == len(paths):
            rows.extend(camera_rows)
    return rows


def frame_details(db, root, camera, path, annotation, settings):
    config = scan_signature(settings, annotation[1])
    row = db.execute("SELECT details FROM frames WHERE root=? AND camera=? AND config=? AND path=?",
                     (str(root), camera, config, path)).fetchone()
    return json.loads(row[0])["detections"]

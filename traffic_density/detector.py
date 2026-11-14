"""YOLO26m detection adapted from app 3.py."""
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou

# COCO class IDs. Persons and bicycles are evidence only.
GROUP_OF = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "car", 7: "car"}
GROUPS = ["car", "motorcycle", "person", "bicycle"]

LARGE_AREA_FRAC = 0.04   # box/frame area above which the lower "close-up car" bar applies
EDGE_PAD_FRAC = 0.08     # grey border so vehicles cut by the frame edge look complete
MIN_SIDE = 8             # px
WEAK_FRAC = 0.5          # riders / bikes down to 50% of their bar may confirm each other
RESCUE_CONF = 0.08       # very low bar when re-searching around a rider for the bike under them
MAX_RESCUE = 40          # cap on rider crops re-examined per frame
MAX_RIDERS = 3
MIN_VOTES = 3            # a faint, rider-less motorcycle is kept if this many passes agree on it
DUP_IOU = 0.40          # two motorcycle boxes overlapping this much are the same bike
DUP_CONTAIN = 0.70       # ...or when this much of the smaller box sits inside the larger one
SEATED_RATIO = 0.62      # w/h of a person box above which the person looks seated


def enhance_night_frame(img: np.ndarray) -> np.ndarray:
    """Glare suppression + adaptive gamma + CLAHE + light denoise, for dark frames only."""
    l_ch, a_ch, b_ch = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    mean = float(l_ch.mean())
    if mean >= 90:
        return img

    _, glare = cv2.threshold(l_ch, 220, 255, cv2.THRESH_BINARY)
    glare = cv2.GaussianBlur(glare, (15, 15), 0).astype(np.float32) / 255.0
    l_ch = np.clip(l_ch * (1.0 - 0.3 * glare), 0, 255).astype(np.uint8)

    gamma = float(np.clip(1.2 + (90 - mean) / 90 * 0.8, 1.2, 2.0))
    lut = (((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255).astype(np.uint8)
    l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.LUT(l_ch, lut))

    out = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    return cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)


# ==========================================
# 3. YOLO PASSES (full frame, reduced size, tiles)
# ==========================================
def predict(model, img, conf, imgsz, classes, pad_frac=0.0):
    """Run YOLO (optionally on a grey-padded copy). Returns boxes, scores, group ids."""
    h, w = img.shape[:2]
    py, px = int(h * pad_frac), int(w * pad_frac)
    src = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=(114, 114, 114)) if pad_frac else img

    r = model.predict(source=src, classes=classes, conf=conf, imgsz=imgsz, iou=0.5, verbose=False)[0]
    if len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)

    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32) - np.array([px, py, px, py], np.float32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    scores = r.boxes.conf.cpu().numpy().astype(np.float32)
    groups = np.array([GROUPS.index(GROUP_OF[int(c)]) for c in r.boxes.cls.cpu().numpy()])
    ok = ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)
    return boxes[ok], scores[ok], groups[ok]


def tile_origins(size: int, tile: int, overlap: float = 0.25) -> List[int]:
    if size <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, size - tile + 1, step))
    return starts if starts[-1] == size - tile else starts + [size - tile]


def tile_pass(model, img, conf, imgsz, classes, tile, keep_groups=None) -> list:
    """Overlapping tiles; boxes clipped by an interior tile edge are dropped as fragments."""
    h, w = img.shape[:2]
    parts = []
    for y0 in tile_origins(h, tile):
        for x0 in tile_origins(w, tile):
            crop = img[y0:y0 + tile, x0:x0 + tile]
            ch, cw = crop.shape[:2]
            b, s, g = predict(model, crop, conf, imgsz, classes)
            if len(b) == 0:
                continue
            cut = np.zeros(len(b), bool)
            if x0 > 0:
                cut |= b[:, 0] <= 2
            if x0 + cw < w:
                cut |= b[:, 2] >= cw - 2
            if y0 > 0:
                cut |= b[:, 1] <= 2
            if y0 + ch < h:
                cut |= b[:, 3] >= ch - 2
            keep = ~cut
            if keep_groups is not None:
                keep &= np.isin(g, keep_groups)
            parts.append((b[keep] + np.array([x0, y0, x0, y0], np.float32), s[keep], g[keep]))
    return parts


def collect_detections(model, img, conf, imgsz, classes, use_tiling, pad_frac) -> List[dict]:
    h, w = img.shape[:2]
    parts = []

    # Full frame at full and reduced size (reduced sees big close-up vehicles whole)
    for size in sorted({imgsz, 640}, reverse=True):
        parts.append(predict(model, img, conf, size, classes, pad_frac))

    if use_tiling and max(h, w) >= 640:
        # Coarse tiles recover small / distant vehicles of every kind
        parts += tile_pass(model, img, conf, imgsz, classes, int(max(480, min(h, w) * 0.6)))
        # Fine tiles blow up far-away scooters and riders (motorcycle + person only)
        parts += tile_pass(model, img, conf, 640, classes, int(max(256, min(h, w) * 0.35)),
                           keep_groups=[GROUPS.index(g) for g in ("motorcycle", "person", "bicycle")])

    boxes = np.concatenate([p[0] for p in parts])
    scores = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts])
    if len(boxes) == 0:
        return []

    keep = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(groups), 0.5).numpy()
    # votes = how many passes (full frame / tiles) found this same object; a faint box that
    # several passes agree on is far more trustworthy than a faint box seen once
    iou = box_iou(torch.from_numpy(boxes[keep]), torch.from_numpy(boxes)).numpy()
    votes = ((iou >= 0.5) & (groups[keep][:, None] == groups[None, :])).sum(axis=1)
    return [{"name": GROUPS[groups[i]], "conf": float(scores[i]), "box": boxes[i], "votes": int(v)}
            for i, v in zip(keep, votes)]


def vote_vehicle_class(dets: List[dict], iou_thr: float = 0.55) -> List[dict]:
    """Overlapping car/motorcycle boxes vote by summed confidence, so one vehicle gets one label."""
    veh = sorted((d for d in dets if d["name"] != "person"), key=lambda d: -d["conf"])
    others = [d for d in dets if d["name"] == "person"]
    if len(veh) < 2:
        return dets

    boxes = torch.tensor(np.stack([d["box"] for d in veh]), dtype=torch.float32)
    iou = box_iou(boxes, boxes).numpy()
    used = np.zeros(len(veh), bool)
    kept = []
    for i in range(len(veh)):
        if used[i]:
            continue
        members = [j for j in range(len(veh)) if not used[j] and iou[i, j] >= iou_thr]
        used[members] = True
        votes: Dict[str, float] = {}
        for j in members:
            votes[veh[j]["name"]] = votes.get(veh[j]["name"], 0.0) + veh[j]["conf"]
        winner = max(votes, key=votes.get)
        kept.append(max((veh[j] for j in members if veh[j]["name"] == winner), key=lambda d: d["conf"]))
    return others + kept


def plausible(name: str, box: np.ndarray) -> bool:
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    if w < MIN_SIDE or h < MIN_SIDE:
        return False
    ratio = w / h
    if name == "motorcycle":
        return 0.25 <= ratio <= 3.0
    if name == "person":
        return 0.15 <= ratio <= 1.6
    return True


# ==========================================
# 4. RIDERS: persons confirm / create motorcycles
# ==========================================
def rider_fit(p: np.ndarray, m: np.ndarray) -> float:
    """0 if the person is not riding this motorcycle box, else a score (higher = better fit)."""
    pw, ph, mw, mh = p[2] - p[0], p[3] - p[1], m[2] - m[0], m[3] - m[1]
    if min(pw, ph, mw, mh) <= 0 or not 0.5 <= ph / mh <= 4.0:
        return 0.0

    inter = max(0.0, min(p[2], m[2]) - max(p[0], m[0])) * max(0.0, min(p[3], m[3]) - max(p[1], m[1]))
    overlap = inter / (pw * ph)
    if overlap >= 0.30:
        return 1.0 + overlap

    cx, hip_y, feet_y = (p[0] + p[2]) / 2, p[1] + 0.70 * ph, p[3]
    x_ok = m[0] - 0.25 * mw <= cx <= m[2] + 0.25 * mw
    hip_ok = m[1] - 1.3 * mh <= hip_y <= m[3]
    feet_ok = m[1] + 0.25 * mh <= feet_y <= m[3] + 0.5 * mh
    if x_ok and hip_ok and feet_ok:
        return 0.5 + 0.5 * (1.0 - min(1.0, abs(cx - (m[0] + m[2]) / 2) / mw))
    return 0.0


def stacked(a: np.ndarray, b: np.ndarray) -> bool:
    """Two person boxes on top of each other, like rider + pillion."""
    aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    if min(aw, ah, bw, bh) <= 0:
        return False
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / min(aw, bw)
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / min(ah, bh)
    return ox >= 0.45 and oy >= 0.40 and abs(a[3] - b[3]) <= 0.30 * max(ah, bh)


def seated(b: np.ndarray) -> bool:
    w, h = b[2] - b[0], b[3] - b[1]
    return h >= 20 and w / h >= SEATED_RATIO


def infer_missed_bikes(people: List[dict], img_h: int) -> Tuple[List[dict], set]:
    """Persons that look like they ride a bike the detector missed -> one inferred motorcycle."""
    parent = list(range(len(people)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            if stacked(people[i]["box"], people[j]["box"]):
                parent[find(i)] = find(j)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(people)):
        clusters.setdefault(find(i), []).append(i)

    bikes, used = [], set()
    for idxs in clusters.values():
        if len(idxs) > MAX_RIDERS or not (len(idxs) >= 2 or seated(people[idxs[0]]["box"])):
            continue
        grp = [people[i] for i in idxs]
        x1, y1 = min(g["box"][0] for g in grp), min(g["box"][1] for g in grp)
        x2, y2 = max(g["box"][2] for g in grp), max(g["box"][3] for g in grp)
        y2 = min(img_h - 1, y2 + 0.30 * max(g["box"][3] - g["box"][1] for g in grp))
        bikes.append({
            "name": "motorcycle", "conf": float(np.mean([g["conf"] for g in grp])) * 0.9,
            "box": np.array([x1, y1, x2, y2], np.float32), "riders": len(grp), "source": "inferred",
        })
        used.update(idxs)
    return bikes, used


def inside(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box a that lies inside box b."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area > 0 else 0.0


def rescue_bikes(model, img, persons, motos) -> List[dict]:
    """
    For every person with no motorcycle box under them, zoom into the area around the person
    and search again at a very low bar. This finds scooters hidden behind / in front of cars
    at a red light, which the whole-frame pass scores too low to keep.
    """
    h, w = img.shape[:2]
    classes = [3]
    found = []
    lone = [p for p in persons if not any(rider_fit(p["box"], m["box"]) > 0 for m in motos)]
    for p in sorted(lone, key=lambda d: -d["conf"])[:MAX_RESCUE]:
        x1, y1, x2, y2 = p["box"]
        pw, ph = x2 - x1, y2 - y1
        cx1, cy1 = int(max(0, x1 - 0.8 * pw)), int(max(0, y1 - 0.2 * ph))
        cx2, cy2 = int(min(w, x2 + 0.8 * pw)), int(min(h, y2 + 0.6 * ph))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.shape[0] < 16 or crop.shape[1] < 16:
            continue
        b, s, _ = predict(model, crop, RESCUE_CONF, 640, classes)
        best = None
        for bx, sc in zip(b + np.array([cx1, cy1, cx1, cy1], np.float32), s):
            fit = rider_fit(p["box"], bx)
            if fit > 0 and (best is None or fit * sc > best[0]):
                best = (fit * sc, {"name": "motorcycle", "conf": float(sc), "box": bx.astype(np.float32),
                                   "source": "rescued"})
        if best:
            found.append(best[1])
    return found


def reclaim_riders_from_cars(cars: List[dict], persons: List[dict]) -> Tuple[List[dict], List[dict]]:
    """A tall, narrow 'car' box with a person inside is a scooter and rider, not a car."""
    keep, moved = [], []
    for c in cars:
        w, h = c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]
        if w / h < 0.6 and any(inside(p["box"], c["box"]) >= 0.5 for p in persons):
            moved.append({**c, "name": "motorcycle"})
        else:
            keep.append(c)
    return keep, moved


def resolve_motorcycles(persons, motos, p_bar, m_bar, infer_bikes, img_h) -> Tuple[List[dict], set]:
    """Pair persons with motorcycle boxes; weak bikes survive only if a rider confirms them."""
    riders_of: Dict[int, List[int]] = {i: [] for i in range(len(motos))}
    matched = set()

    pairs = sorted(
        ((rider_fit(p["box"], m["box"]), pi, mi)
         for pi, p in enumerate(persons) for mi, m in enumerate(motos)),
        reverse=True,
    )
    for score, pi, mi in pairs:
        if score > 0 and pi not in matched and len(riders_of[mi]) < MAX_RIDERS:
            riders_of[mi].append(pi)
            matched.add(pi)

    out = []
    for mi, m in enumerate(motos):
        riders = [persons[i] for i in riders_of[mi]]
        if m["conf"] < m_bar and not riders and m.get("votes", 1) < MIN_VOTES:
            continue  # faint, no rider, and not confirmed by several passes
        box = m["box"].copy()
        for r in riders:  # stretch the box to cover its riders
            box[:2] = np.minimum(box[:2], r["box"][:2])
            box[2:] = np.maximum(box[2:], r["box"][2:])
        out.append({**m, "box": box, "region_box": m["box"], "riders": len(riders), "source": m.get("source", "detected")})

    if infer_bikes:
        # A person already sitting on a known bike (e.g. a pillion) never creates another one
        free = [p for pi, p in enumerate(persons)
                if pi not in matched and p["conf"] >= p_bar
                and not any(inside(p["box"], m["box"]) >= 0.5 for m in out)]
        bikes, _ = infer_missed_bikes(free, img_h)
        out += bikes

    return merge_motorcycles(out), matched


def overlap_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(IoU, intersection / smaller box area) of two boxes."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-6), inter / max(min(aa, ab), 1e-6)


def merge_motorcycles(bikes: List[dict]) -> List[dict]:
    """
    One physical motorcycle = one box. Boxes of the same bike that come from different steps
    (bike box, rider+bike box, rescued bike, inferred bike) overlap heavily or sit inside one
    another, so they are fused into the most confident one with a box covering them all.
    """
    kept: List[dict] = []
    for d in sorted(bikes, key=lambda x: -x["conf"]):
        for k in kept:
            iou, contain = overlap_stats(d["box"], k["box"])
            if iou >= DUP_IOU or contain >= DUP_CONTAIN:
                k["box"] = np.concatenate([np.minimum(k["box"][:2], d["box"][:2]),
                                           np.maximum(k["box"][2:], d["box"][2:])])
                k["riders"] = min(MAX_RIDERS, max(k.get("riders", 0), d.get("riders", 0)))
                break
        else:
            kept.append({**d})
    return kept


def detect(model, original, *, imgsz=1280, use_tiling=True, night_mode=True,
           pad_edges=True, infer_bikes=True, car_bar=0.22, large_car_bar=0.12,
           moto_bar=0.15, rider_bar=0.20):
    """Return vehicles and pedestrian candidates in coordinates of the input PIL image."""
    img = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
    shown = enhance_night_frame(img) if night_mode else img
    # Low-resolution CCTV frames are upscaled first so distant riders are big enough to detect
    # (and so tiling is never skipped); boxes are mapped back to the original size at the end.
    scale = max(1.0, imgsz / max(shown.shape[:2]))
    proc = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else shown
    h, w = proc.shape[:2]
    classes = list(GROUP_OF)

    # Run at the lowest bar anywhere; the per-class bars are applied afterwards
    base_conf = min(car_bar, large_car_bar, moto_bar * WEAK_FRAC, rider_bar * WEAK_FRAC)
    dets = collect_detections(model, proc, base_conf, imgsz, classes, use_tiling,
                              EDGE_PAD_FRAC if pad_edges else 0.0)
    # Keep bicycle evidence to stop a cyclist from being inferred as a motorcycle.
    bicycles = [d for d in dets if d["name"] == "bicycle" and d["conf"] >= moto_bar]
    dets = vote_vehicle_class([
        d for d in dets
        if d["name"] != "bicycle"
        and (d["name"] != "motorcycle" or not any(
            b["conf"] >= d["conf"] and overlap_stats(b["box"], d["box"])[0] >= 0.50
            for b in bicycles
        ))
    ])

    cars, motos, persons = [], [], []
    for d in dets:
        if not plausible(d["name"], d["box"]):
            continue
        if d["name"] == "car":
            area = (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
            bar = large_car_bar if area / (h * w) >= LARGE_AREA_FRAC else car_bar
            if d["conf"] >= bar:
                cars.append(d)
        elif d["name"] == "motorcycle" and d["conf"] >= moto_bar * WEAK_FRAC:
            motos.append(d)
        elif d["name"] == "person" and d["conf"] >= rider_bar * WEAK_FRAC:
            persons.append(d)

    persons = [p for p in persons if not any(
        rider_fit(p["box"], b["box"]) > 0
        and not any(
            m["conf"] > b["conf"] and rider_fit(p["box"], m["box"]) > 0
            for m in motos
        )
        for b in bicycles
    )]
    cars, reclaimed = reclaim_riders_from_cars(cars, persons)
    motos += reclaimed
    motos += rescue_bikes(model, proc, persons, motos)
    motos, matched_riders = resolve_motorcycles(persons, motos, rider_bar, moto_bar, infer_bikes, h)
    pedestrians = [p for i, p in enumerate(persons)
                   if i not in matched_riders and p["conf"] >= rider_bar]
    # A vehicle is either a car or a motorcycle: drop a car box sitting on a motorcycle box
    cars = [c for c in cars if all(overlap_stats(c["box"], m["box"])[0] < 0.5 for m in motos)]
    final = sorted(cars + motos + pedestrians, key=lambda d: d["box"][1])
    scale_xy = np.array([w / img.shape[1], h / img.shape[0]] * 2)
    return [
        {
            "name": d["name"],
            "conf": d["conf"],
            "box": (d["box"] / scale_xy).tolist(),
            "region_box": (d.get("region_box", d["box"]) / scale_xy).tolist(),
            "riders": d.get("riders", 0),
            "source": d.get("source", "detected"),
        }
        for d in final
    ]

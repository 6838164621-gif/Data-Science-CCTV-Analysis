"""Streamlit dashboard for calculating and inspecting cleaned CCTV traffic density."""
import csv
import os
from datetime import datetime, time, timedelta
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import time_ns

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from dataset import (
    annotations, cache, count_dataset, discover, file_digest, fingerprint,
    frame_details, report, timestamp,
)
from regions import load_labelme_regions, read_annotation, region_overlay

APP = Path(__file__).resolve().parent
ORIGINAL_IMAGES = Path("cctv_cleaner/automatic_results/cleaned_dataset/images")
DESKTOP = Path.home() / "Desktop" / "Cleaned_CCTV"
DEFAULT_IMAGES = next((p for p in (APP / ORIGINAL_IMAGES, DESKTOP / ORIGINAL_IMAGES,
                                    DESKTOP / "cleaned_cctv_images") if p.is_dir()),
                      APP / ORIGINAL_IMAGES)
PROJECT_LABELS = APP / "Data" / "Labelme workflow" / "drafts"
DEFAULT_LABELS = next((p for p in (PROJECT_LABELS, Path.home() / "Downloads/Labelme workflow/drafts")
                       if p.is_dir()), PROJECT_LABELS)
DEFAULT_INFO = next((p for p in (APP / "Data/cctv_info.csv",
                                 Path.home() / "Downloads/cctv_info.csv") if p.is_file()),
                    APP / "Data/cctv_info.csv")
BUNDLED_TEST_WEIGHTS = APP.parent.parent / "work/yolo26m.pt"
COLORS = {"car": "#67e8a0", "motorcycle": "#ffb66b", "person": "#70f5ff"}


@st.cache_data(show_spinner=False)
def inventory(images, labels, scan):
    cameras = discover(images)
    return cameras, *annotations(labels, cameras)


@st.cache_data(show_spinner=False)
def checksum(path, file_signature):
    return file_digest(path)


def camera_names(info):
    if not info.is_file():
        return {}
    try:
        with info.open(encoding="utf-8-sig", newline="") as stream:
            return {row["camera_id"].strip(): row.get("location_en") or row.get("location", "")
                    for row in csv.DictReader(stream)}
    except (OSError, UnicodeError, KeyError, csv.Error) as error:
        st.warning(f"Camera names could not be loaded. {error}")
        return {}


def show_frame(db, root, row, labels, settings):
    path = root / row["image"]
    with Image.open(path) as original:
        image = original.convert("RGB")
    annotation = labels[row["camera_id"]]
    detections = frame_details(db, root, row["camera_id"], row["image"], annotation, settings)
    if st.checkbox("Show LabelMe regions", value=True):
        image = region_overlay(image, load_labelme_regions(annotation[0], annotation[2], image.size))
        image = region_overlay(image, load_labelme_regions(
            annotation[0], annotation[2], image.size, sidewalk=True,
        ))
        st.caption("Blue = road · Pink = crosswalk · Orange = highway · Green = sidewalk")
    draw = ImageDraw.Draw(image)
    for item in detections:
        color = COLORS[item["name"]]
        draw.rectangle(item["box"], outline=color, width=2)
        x1, _, x2, y2 = item["region_box"]
        x = int((x1 + x2) / 2)
        draw.ellipse((x - 3, y2 - 3, x + 3, y2 + 3), fill=color)
    st.image(image, width="stretch")
    columns = st.columns(3)
    columns[0].metric("Road + crosswalk", f'{row["road_density_pct"]:.1f}%',
                      help="Share of the camera’s maximum road and crosswalk count combined.")
    columns[0].caption(f'{row["road_count"]} vehicles · maximum {row["road_max"]}')
    columns[1].metric("Highway", f'{row["highway_density_pct"]:.1f}%')
    columns[1].caption(f'{row["highway_count"]} vehicles · maximum {row["highway_max"]}')
    if pd.notna(row["sidewalk_pedestrian_count"]):
        columns[2].metric("Sidewalk pedestrians", f'{row["sidewalk_pedestrian_density_pct"]:.1f}%')
        columns[2].caption(
            f'{int(row["sidewalk_pedestrian_count"])} people · maximum {int(row["sidewalk_pedestrian_max"])}'
        )
    else:
        columns[2].metric("Sidewalk pedestrians", "Unavailable")
    st.dataframe([{"Region": r.title(), "Cars, buses and trucks": row[f"{r}_car"],
                   "Motorcycles": row[f"{r}_motorcycle"]}
                  for r in ("road", "crosswalk", "highway", "background")], hide_index=True)
    if detections:
        with st.expander("Vehicle and pedestrian detections"):
            st.dataframe([
                {"Class": d["name"], "Region": d["region"], "Confidence (%)": round(100 * d["conf"], 1),
                 "Riders": d["riders"], "Source": d["source"],
                 "Box (x1, y1, x2, y2)": ", ".join(str(round(n, 1)) for n in d["box"])}
                for d in detections
            ], hide_index=True)


def period_controls(camera, paths):
    dates = [captured[:10] for path in paths if (captured := timestamp(path))]
    if not dates:
        st.caption("Time filtering requires capture timestamps in the image filenames.")
        return None
    first = datetime.fromisoformat(min(dates)).date()
    last = datetime.fromisoformat(max(dates)).date()
    key = f"{camera}_{first}_{last}"
    start_col, end_col = st.columns(2)
    start_date = start_col.date_input("Start date", first, min_value=first, max_value=last,
                                      key=f"start_date_{key}")
    start_time = start_col.time_input("Start time", time.min, step=60, key=f"start_time_{key}")
    end_date = end_col.date_input("End date", last, min_value=first, max_value=last,
                                  key=f"end_date_{key}")
    end_time = end_col.time_input("End time", time(23, 59), step=60, key=f"end_time_{key}",
                                  help="Includes every frame captured during this minute.")
    start, end = datetime.combine(start_date, start_time), datetime.combine(end_date, end_time)
    if start > end:
        st.warning("Choose an end date and time at or after the start.")
        return None
    return start, end + timedelta(minutes=1)


def hourly_density(table, column):
    data = table[["captured_at", column]].dropna()
    hours = pd.to_datetime(data.captured_at, errors="coerce").dt.floor("h")
    result = data.groupby(hours)[column].agg(density="mean", frames="size").reset_index()
    result = result.rename(columns={"captured_at": "hour"})
    minimum, maximum = result.density.min(), result.density.max()
    result["level"] = "Other hours"
    if minimum < maximum:
        result.loc[result.density.eq(maximum), "level"] = "Highest"
        result.loc[result.density.eq(minimum), "level"] = "Lowest"
    return result


def hourly_chart(table, column, title):
    hourly = hourly_density(table, column)
    if hourly.empty:
        return
    hourly["end"] = hourly.hour + pd.Timedelta(hours=1)
    hourly["hour_label"] = hourly.hour.dt.strftime("%d %b %Y, %H:%M")
    for field in ("hour", "end"):
        # Encode the camera's local clock on a UTC scale to preserve its labels in the browser.
        hourly[field] = hourly[field].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    x = alt.X("hour:T", title="Capture hour (local time)", scale=alt.Scale(type="utc", nice=False))
    chart = alt.Chart(alt.Data(values=hourly.to_dict("records"))).encode(
        x=x, x2="end:T",
        color=alt.Color("level:N", title=None, scale=alt.Scale(
            domain=["Highest", "Lowest", "Other hours"], range=["#b84432", "#147b70", "#6586b4"])),
        tooltip=[alt.Tooltip("hour_label:N", title="Hour starting"),
                 alt.Tooltip("density:Q", title="Mean density (%)", format=".2f"),
                 alt.Tooltip("frames:Q", title="Frames"), alt.Tooltip("level:N", title="Rank")],
    )
    y = alt.Y("density:Q", title="Mean density (%)", scale=alt.Scale(domain=[0, 100]))
    bars = chart.mark_bar(clip=True).encode(y=y, y2=alt.Y2Datum(0))
    # A colored baseline tick keeps true 0% hours visible.
    zeros = chart.transform_filter(alt.datum.density == 0).mark_rule(strokeWidth=3).encode(y=y)
    st.altair_chart(alt.layer(bars, zeros).properties(title=title, height=240).interactive(bind_y=False),
                    width="stretch")
    st.caption(f"Highest hourly density {hourly.density.max():.2f}% · "
               f"Lowest hourly density {hourly.density.min():.2f}%")


def dashboard(rows, camera, period, labels, settings, db, root):
    table = pd.DataFrame(rows).sort_values(["captured_at", "image"])
    st.download_button("Download full angle CSV", table.to_csv(index=False).encode("utf-8-sig"),
                       f"traffic_density_{camera}.csv", "text/csv")
    if period is None and table.captured_at.ne("").any():
        return
    filtered = table
    if period:
        start, end = (point.isoformat(sep=" ") for point in period)
        filtered = table[table.captured_at.ge(start) & table.captured_at.lt(end)]
    if filtered.empty:
        st.info("No cleaned frames were captured in this period.")
        return
    st.subheader("Density in the selected period")
    st.caption("Each bar averages the selected frames within one hour. Percentages use the "
               f"baseline counts from all {len(table):,} cleaned frames for camera {camera}.")
    hourly_chart(filtered, "road_density_pct", "Road + crosswalk vehicles")
    annotation = read_annotation(labels[camera][0])
    if any(str(shape.get("label", "")).strip().lower() == "highway"
           and len(shape.get("points", [])) >= 3 for shape in annotation["shapes"]):
        hourly_chart(filtered, "highway_density_pct", "Highway vehicles")
    st.subheader("Sidewalk pedestrian density")
    if filtered["sidewalk_pedestrian_max"].notna().any():
        hourly_chart(filtered, "sidewalk_pedestrian_density_pct", "Sidewalk pedestrians")
    else:
        st.info("This camera angle needs a LabelMe sidewalk polygon to show pedestrian density.")
    st.subheader("Frames in the selected period")
    st.dataframe(filtered, hide_index=True, column_config={
        "road_density_pct": st.column_config.NumberColumn("Road density (%)", format="%.2f"),
        "highway_density_pct": st.column_config.NumberColumn("Highway density (%)", format="%.2f"),
        "sidewalk_pedestrian_density_pct": st.column_config.NumberColumn("Sidewalk density (%)", format="%.2f"),
    })
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8-sig"),
                       f"traffic_density_{camera}_period.csv", "text/csv")
    path = st.selectbox("Inspect frame", filtered.image)
    row = filtered.loc[filtered.image == path].iloc[0]
    try:
        show_frame(db, root, row, labels, settings)
    except (OSError, ValueError) as error:
        st.error(f"Could not open the saved frame. Refresh the inventory. {error}")


def run_scan(db, root, camera, paths, annotation, settings, weights):
    total = len(paths)
    bar = st.progress(0.0, text=f"Checking {total:,} cleaned frames for camera {camera}")
    done = 0
    detect_image = None

    def progress():
        nonlocal done
        done += 1
        if done % max(1, total // 300) == 0 or done == total:
            bar.progress(done / total, text=f"{done:,} / {total:,} frames checked")

    def predict(image):
        nonlocal detect_image
        if detect_image is None:
            from ultralytics import YOLO
            from detector import detect

            model = YOLO(str(weights))
            if settings["device"] != "auto":
                model.to(settings["device"])
            detect_image = partial(detect, model, **settings["detection"])
        return detect_image(image)

    errors = count_dataset(db, root, {camera: paths}, {camera: annotation}, settings, predict, progress)
    if errors:
        st.warning("This angle has frames or annotations that need attention.")
        st.dataframe([{"Path": path, "Issue": error} for path, error in errors[camera]],
                     hide_index=True)


def main():
    st.set_page_config(page_title="CCTV traffic density", layout="wide")
    st.title("CCTV traffic density")
    st.caption("Vehicle and sidewalk pedestrian densities use separate maximum counts across all cleaned frames for the selected angle.")
    with st.sidebar:
        st.header("Data")
        root = Path(st.text_input("Cleaned image folder", os.getenv("CCTV_IMAGES", str(DEFAULT_IMAGES)))).expanduser().resolve()
        label_root = Path(st.text_input("LabelMe drafts folder", os.getenv("CCTV_LABELME", str(DEFAULT_LABELS)))).expanduser().resolve()
        info = Path(st.text_input("Camera metadata CSV (optional)", str(DEFAULT_INFO))).expanduser()
        cache_path = Path(st.text_input("Results database", str(APP / "traffic_density.sqlite3"))).expanduser()
        default_weight = APP / "yolo26m.pt"
        if not default_weight.is_file() and BUNDLED_TEST_WEIGHTS.is_file():
            default_weight = BUNDLED_TEST_WEIGHTS
        weights = Path(st.text_input("YOLO26m weights", os.getenv("CCTV_WEIGHTS", str(default_weight)))).expanduser().resolve()
        with st.expander("Detection settings"):
            settings = {"detection": {
                "imgsz": st.select_slider("Inference size", [640, 960, 1280, 1536, 1920], 1280),
                "use_tiling": st.checkbox("Tiled inference", True),
                "night_mode": st.checkbox("Low-light enhancement", True),
                "pad_edges": st.checkbox("Edge padding", True),
                "infer_bikes": st.checkbox("Infer motorcycles from rider evidence", True),
                "car_bar": st.slider("Car confidence", 0.05, 0.9, 0.22),
                "large_car_bar": st.slider("Close-up car confidence", 0.05, 0.6, 0.12),
                "moto_bar": st.slider("Motorcycle confidence", 0.05, 0.9, 0.15),
                "rider_bar": st.slider("Rider confidence", 0.05, 0.9, 0.20),
            }, "device": st.selectbox("Device", ["auto", "cpu", "mps", "cuda:0"])}
        refresh = st.button("Refresh inventory")
    if not root.is_dir():
        st.error("Choose an existing cleaned image folder. Extract the ZIP first if needed.")
        return
    if refresh or "inventory_version" not in st.session_state:
        st.session_state.inventory_version = time_ns()

    try:
        cameras, labels, problems = inventory(root, label_root, st.session_state.inventory_version)
        if not cameras:
            st.warning("No cleaned images were found in this folder.")
            return
        names = camera_names(info)
        st.write(f"{sum(map(len, cameras.values())):,} cleaned images · {len(cameras)} cameras · "
                 f"{len(labels)} cameras with LabelMe regions")
        if problems:
            with st.expander(f"{len(problems)} cameras need LabelMe annotations", expanded=not labels):
                st.caption("Place one JSON per camera under the drafts folder. The filename must match the camera ID.")
                st.dataframe([{"Camera": cid, "Location": names.get(cid, ""), "Images": len(cameras[cid]),
                               "Issue": issue} for cid, issue in problems.items()], hide_index=True)
        if not labels:
            return
        camera = st.selectbox("Camera angle", list(labels), format_func=lambda cid:
                              f"{cid} · {names[cid]}" if names.get(cid) else cid)
        paths = cameras[camera]
        st.caption(f"{len(paths):,} cleaned frames for this angle. The time range filters the frame results.")
        period = period_controls(camera, paths)
        run = st.button("Process this camera angle", type="primary")
        if run:
            st.session_state.inventory_version = time_ns()
            cameras, labels, _ = inventory(root, label_root, st.session_state.inventory_version)
            if camera not in labels:
                st.warning("This camera's images or annotation changed. Refresh the inventory.")
                return
            paths = cameras[camera]
        if not weights.is_file():
            st.warning("Select a local yolo26m.pt file. See the setup instructions for downloading the weights.")
            return
        settings["weights"] = checksum(weights, fingerprint(weights))
        settings["pipeline"] = {name: file_digest(APP / name)
                                for name in ("detector.py", "regions.py", "dataset.py")}
        settings["libraries"] = {}
        for package in ("ultralytics", "torch", "torchvision", "opencv-python", "pillow", "numpy"):
            try:
                settings["libraries"][package] = version(package)
            except PackageNotFoundError:
                settings["libraries"][package] = "unavailable"
        with cache(cache_path) as db:
            if run:
                try:
                    run_scan(db, root, camera, paths, labels[camera], settings, weights)
                except Exception as error:
                    st.error(f"Processing stopped. Saved frames will be reused when you retry. {error}")
                    st.stop()
                # Confirm the current inventory after a potentially long run.
                st.session_state.inventory_version = time_ns()
                cameras, labels, _ = inventory(root, label_root, st.session_state.inventory_version)
                if camera not in labels:
                    st.warning("This camera's images or annotation changed. Refresh the inventory.")
                    return
            rows = report(db, root, {camera: cameras[camera]}, labels, settings)
            if rows:
                dashboard(rows, camera, period, labels, settings, db, root)
            else:
                st.info("Process this angle to calculate its maxima and density results.")
    except (OSError, ValueError) as error:
        st.error(str(error))


if __name__ == "__main__":
    main()

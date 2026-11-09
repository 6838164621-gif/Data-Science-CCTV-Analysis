from pathlib import Path
import re

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
INFO_PATH = BASE_DIR / "Data" / "cctv_info.csv" # make sure to replace with your own folder (i put the csv and cctv_images folder inside of a Data folder)
IMAGE_ROOT = BASE_DIR / "Data" / "cctv_images" / "cctv_images"


@st.cache_data
def load_data():
    info = pd.read_csv(INFO_PATH, dtype={"camera_id": str})
    info["camera_id"] = info["camera_id"].str.strip()
    rows = []
    pattern = re.compile(r"_(\d{8})_(\d{6})\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    for path in IMAGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if not match:
            continue
        camera_id = path.parent.name
        stamp = pd.to_datetime(f"{match.group(1)} {match.group(2)}", format="%Y%m%d %H%M%S", errors="coerce")
        if pd.notna(stamp):
            rows.append({"camera_id": camera_id, "captured_at": stamp, "image_path": str(path)})
    images = pd.DataFrame(rows)
    if images.empty:
        return pd.DataFrame()
    data = images.merge(info, on="camera_id", how="left")
    data["date"] = data["captured_at"].dt.date
    data["time"] = data["captured_at"].dt.time
    data["time_minutes"] = data["captured_at"].dt.hour * 60 + data["captured_at"].dt.minute
    data["camera_label"] = data.apply(
        lambda r: f"{r['camera_id']} — {r.get('location_en', '')}" if pd.notna(r.get('location_en')) else str(r['camera_id']),
        axis=1,
    )
    return data.sort_values("captured_at").reset_index(drop=True)


st.set_page_config(page_title="CCTV Date, Time & Location Explorer", page_icon="📹", layout="wide")
st.title("📹 CCTV Date, Time & Location Explorer")
st.caption("Explore the dates and times available for each camera angle, with location details from the CCTV metadata.")

data = load_data()
if data.empty:
    st.error("No timestamped CCTV images were found in the Data folder.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    camera_options = (data[["camera_id", "camera_label"]]
                      .drop_duplicates()
                      .sort_values("camera_id", key=lambda s: pd.to_numeric(s, errors="coerce"))["camera_label"]
                      .tolist())
    selected_cameras = st.multiselect("Camera angle", camera_options, placeholder="All camera angles")

    min_date, max_date = data["date"].min(), data["date"].max()
    date_range = st.date_input("Capture date", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    time_range = st.slider("Time of day", min_value=0, max_value=1439, value=(0, 1439), step=1,
                           format="%H:%M")
    st.divider()
    st.info("Select one or more camera angles, a date range, and a time range. Results update automatically.")

filtered = data[
    data["date"].between(start_date, end_date)
    & data["time_minutes"].between(time_range[0], time_range[1])
]
if selected_cameras:
    filtered = filtered[filtered["camera_label"].isin(selected_cameras)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Matching captures", f"{len(filtered):,}")
col2.metric("Camera angles", f"{filtered['camera_id'].nunique():,}")
col3.metric("Dates", f"{filtered['date'].nunique():,}")
col4.metric("Locations", f"{filtered['location_en'].nunique():,}")

if filtered.empty:
    st.warning("No captures match the selected filters.")
    st.stop()

st.subheader("Availability by camera angle")
summary = (filtered.groupby(["camera_id", "location_en", "location_detail_en"], dropna=False)
           .agg(captures=("image_path", "size"), first_capture=("captured_at", "min"), last_capture=("captured_at", "max"))
           .reset_index())
summary["camera_sort"] = pd.to_numeric(summary["camera_id"], errors="coerce")
summary = summary.sort_values("camera_sort").drop(columns="camera_sort")
summary["first_capture"] = summary["first_capture"].dt.strftime("%Y-%m-%d %H:%M:%S")
summary["last_capture"] = summary["last_capture"].dt.strftime("%Y-%m-%d %H:%M:%S")
summary.columns = ["Camera ID", "Location", "Direction / detail", "Captures", "First capture", "Last capture"]
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Capture frequency")
st.caption("These histograms use the currently filtered captures and are grouped by CCTV camera ID.")
frequency_tab_date, frequency_tab_time = st.tabs(["By date", "By time of day"])

with frequency_tab_date:
    date_frequency = (filtered.groupby(["date", "camera_id"], as_index=False)
                      .size().rename(columns={"size": "captures"}))
    date_frequency["camera_id"] = pd.Categorical(
        date_frequency["camera_id"],
        categories=sorted(filtered["camera_id"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x)),
        ordered=True,
    )
    date_chart = date_frequency.pivot(index="date", columns="camera_id", values="captures").fillna(0)
    st.bar_chart(date_chart, height=400)

with frequency_tab_time:
    filtered = filtered.copy()
    filtered["hour"] = filtered["captured_at"].dt.hour
    time_frequency = (filtered.groupby(["hour", "camera_id"], as_index=False)
                      .size().rename(columns={"size": "captures"}))
    time_frequency["camera_id"] = pd.Categorical(
        time_frequency["camera_id"],
        categories=sorted(filtered["camera_id"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x)),
        ordered=True,
    )
    time_chart = time_frequency.pivot(index="hour", columns="camera_id", values="captures").fillna(0)
    time_chart.index = [f"{hour:02d}:00" for hour in time_chart.index]
    st.bar_chart(time_chart, height=400)

st.subheader("Preview")
selected_row = st.selectbox("Choose a capture to preview", filtered.index.tolist(), format_func=lambda i: (
    f"Camera {filtered.loc[i, 'camera_id']} — {filtered.loc[i, 'captured_at']:%Y-%m-%d %H:%M:%S} — {filtered.loc[i, 'location_en']}"
))
row = filtered.loc[selected_row]
left, right = st.columns([1, 1])
with left:
    st.image(row["image_path"], caption=f"Camera {row['camera_id']} | {row['captured_at']:%Y-%m-%d %H:%M:%S}", use_container_width=True)
with right:
    st.write(f"**Location:** {row['location_en']}")
    st.write(f"**Direction / detail:** {row['location_detail_en']}")
    st.write(f"**Coordinates:** {row['latitude']}, {row['longitude']}")


st.subheader("Filtered capture records")
display = filtered[["camera_id", "location_en", "location_detail_en", "captured_at", "latitude", "longitude", "image_path"]].copy()
display.columns = ["Camera ID", "Location", "Direction / detail", "Captured at", "Latitude", "Longitude", "Image path"]
st.dataframe(display, use_container_width=True, hide_index=True, column_config={"Image path": st.column_config.TextColumn(width="large")})

st.subheader("Camera locations")
map_data = filtered[["latitude", "longitude"]].dropna().drop_duplicates()
if not map_data.empty:
    st.map(map_data, zoom=10)

# python -m streamlit run app.py to run website
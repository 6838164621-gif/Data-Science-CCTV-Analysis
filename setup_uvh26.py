from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parent
target = root / "models" / "uvh26"
target.mkdir(parents=True, exist_ok=True)

print("Downloading the UVH-26 YOLOv11-S checkpoint…")
hf_hub_download(
    repo_id="iisc-aim/UVH-26",
    filename="weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt",
    local_dir=str(target),
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id="iisc-aim/UVH-26",
    filename="uvh_classes.txt",
    local_dir=str(target),
    local_dir_use_symlinks=False,
)

downloaded = target / "weights" / "YOLOv11-S" / "UVH-26-MV-YOLOv11-S.pt"
final = target / "UVH-26-MV-YOLOv11-S.pt"
if downloaded.exists() and not final.exists():
    downloaded.replace(final)
print(f"Checkpoint ready: {final}")

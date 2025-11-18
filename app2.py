"""
AI Laundry Sorter – Streamlit Demo
Loads multitask ConvNeXt (color, fabric, wash program)
Checkpoint is downloaded from Google Drive.
"""

import os
import io
import datetime

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import gdown
import streamlit as st

# ------------------------------------------------------------
# 0) Paths + Google Drive download
# ------------------------------------------------------------
LABELS_CSV = "wash_labels.csv"       # this file is in your GitHub repo
HEADER_IMG = "ai.jpg"                # project banner (optional)
CKPT_PATH  = "best_model_wash.pt"    # downloaded here
DEMO_LOG   = "demo_usage_log.csv"

# your Drive link converted to `uc?id=...`
MODEL_URL_WASH = "https://drive.google.com/uc?id=1y5wTHMGzfHasvNMEilpWgNw-A9o9Olmm"


def download_if_missing(url: str, dest: str, what: str = "file"):
    if os.path.exists(dest):
        return
    st.write(f"Downloading {what} from Google Drive...")
    try:
        gdown.download(url, dest, quiet=False)
    except Exception as e:
        st.error(f"Failed to download {what} from Google Drive.\n\n{e}")
        st.stop()


st.write("Downloading wash-model checkpoint from Google Drive...")
download_if_missing(MODEL_URL_WASH, CKPT_PATH, "wash-model checkpoint")

# ------------------------------------------------------------
# 1) Load checkpoint (only fix 'module.' prefix if needed)
# ------------------------------------------------------------
raw_state = torch.load(CKPT_PATH, map_location="cpu")

def strip_module_prefix(state_dict):
    """Remove 'module.' prefix when the model was trained with DataParallel."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state

state_dict = strip_module_prefix(raw_state)

# ------------------------------------------------------------
# 2) Label metadata from CSV  (we use this to know #classes)
# ------------------------------------------------------------
df_all = pd.read_csv(LABELS_CSV)

df_all["color_label"]      = df_all["color_label"].astype(int)
df_all["fabric_label"]     = df_all["fabric_label"].astype(int)
df_all["wash_cycle_label"] = df_all["wash_cycle_label"].astype(int)

num_color_classes  = df_all["color_label"].nunique()
num_fabric_classes = df_all["fabric_label"].nunique()
num_wash_classes   = df_all["wash_cycle_label"].nunique()

color_map  = dict(zip(df_all["color_label"],  df_all["color_group"]))
fabric_map = dict(zip(df_all["fabric_label"], df_all["fabric_group"]))
wash_map   = dict(zip(df_all["wash_cycle_label"], df_all["wash_cycle"]))

wash_full_description = {
    "delicate": "Delicate – gentle wash, cold water, low spin (ideal for silk and sensitive fabrics).",
    "normal":   "Normal – standard wash, 40°C warm water, medium spin (daily cotton & mixed clothes).",
    "heavy":    "Heavy – deep clean, 60°C hot water, high spin (jeans, towels, sportswear).",
    "quick":    "Quick – short cycle, 30°C cool water, medium spin (lightly-soiled clothes).",
    "wool":     "Wool – special wool cycle, 30°C cold, ultra-low spin (prevents shrinkage).",
}

# ------------------------------------------------------------
# 3) Model definition
# ------------------------------------------------------------
IMG_SIZE = 256
BACKBONE_NAME = "convnext_tiny"

demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class WashMultiTaskConvNeXt(nn.Module):
    def __init__(self, num_color: int, num_fabric: int, num_wash: int):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head_color      = nn.Linear(feat_dim, num_color)
        self.head_fabric     = nn.Linear(feat_dim, num_fabric)
        self.head_wash_cycle = nn.Linear(feat_dim, num_wash)

    def forward(self, x):
        feat = self.backbone(x)
        logits_c = self.head_color(feat)
        logits_f = self.head_fabric(feat)
        logits_w = self.head_wash_cycle(feat)
        return logits_c, logits_f, logits_w

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
).to(device)

# load weights (now names should match thanks to strip_module_prefix)
model.load_state_dict(state_dict)
model.eval()

# ------------------------------------------------------------
# 4) Prediction helper
# ------------------------------------------------------------
def predict_single_image(pil_img: Image.Image):
    x = demo_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)

    pc = logits_c.argmax(1).item()
    pf = logits_f.argmax(1).item()
    pw = logits_w.argmax(1).item()

    wash_key = wash_map.get(pw, "normal")
    wash_text = wash_full_description.get(wash_key, wash_key)

    return {
        "color":  color_map.get(pc, "Unknown"),
        "fabric": fabric_map.get(pf, "Unknown"),
        "wash":   wash_text,
    }

# ------------------------------------------------------------
# 5) Streamlit UI
# ------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Laundry Sorter",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("AI Laundry Sorter")
    st.caption("Deep-learning powered wash-program recommendation.")

    uploaded_file = st.file_uploader(
        "Upload a clothing image",
        type=["jpg", "jpeg", "png"],
    )

    if not uploaded_file:
        st.info("Please upload a garment image.")
        return

    pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(pil_img, use_container_width=True)

    result = predict_single_image(pil_img)

    with col2:
        st.subheader("Washing Recommendation")
        st.markdown(f"**Color Group:** {result['color']}")
        st.markdown(f"**Fabric Group:** {result['fabric']}")
        st.markdown(f"**Wash Program:** {result['wash']}")

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    row = pd.DataFrame([{
        "timestamp": ts,
        "image": uploaded_file.name,
        "color": result["color"],
        "fabric": result["fabric"],
        "wash": result["wash"],
    }])
    if os.path.exists(DEMO_LOG):
        old = pd.read_csv(DEMO_LOG)
        pd.concat([old, row], ignore_index=True).to_csv(DEMO_LOG, index=False)
    else:
        row.to_csv(DEMO_LOG, index=False)

    st.success("Prediction logged.")

if __name__ == "__main__":
    main()

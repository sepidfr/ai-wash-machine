"""
AI Laundry Sorter – Streamlit Demo

- Multi-task ConvNeXt-Tiny:
    * Color Group
    * Fabric Group
    * Wash Program
- Uses wash_labels.csv for human-readable labels
- Downloads best_model_wash.pt from Google Drive on first run
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
import streamlit as st
import gdown


# ============================================================
# 0) Paths (relative to repo root)
# ============================================================
HERE = os.path.dirname(__file__)

LABELS_CSV = os.path.join(HERE, "wash_labels.csv")
HEADER_IMG = os.path.join(HERE, "ai.jpg")
DEMO_LOG   = os.path.join(HERE, "demo_usage_log.csv")

CKPT_PATH  = os.path.join(HERE, "best_model_wash.pt")

# ✅ Google Drive link for your wash model checkpoint
# (this is the same ID you استفاده کردی قبل)
MODEL_URL_WASH = (
    "https://drive.google.com/uc?"
    "id=1Siu8S9OVwfu7v13M0ip8ucf4_CRuGUS9"
)


def download_if_missing(url: str, dest_path: str, label: str):
    """Download a file from Google Drive if it does not exist locally."""
    if os.path.exists(dest_path):
        return
    st.info(f"Downloading {label} model from Google Drive (first run only)…")
    gdown.download(url, dest_path, quiet=False)


# ============================================================
# 1) Load label metadata
# ============================================================
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

# Full descriptive wash-cycle names (no numeric labels in UI)
wash_full_description = {
    "delicate": "Delicate – gentle wash, cold water, low spin "
                "(ideal for silk, lace, sensitive fabrics).",
    "normal":   "Normal – standard wash, warm water (~40°C), medium spin "
                "(daily cotton & mixed clothes).",
    "heavy":    "Heavy – deep clean, hot water (~60°C), high spin "
                "(jeans, towels, sportswear).",
    "quick":    "Quick – rapid wash, cool water (~30°C), medium spin "
                "(lightly soiled garments).",
    "wool":     "Wool – special wool cycle, cool water, very low spin "
                "(helps prevent shrinkage).",
}


# ============================================================
# 2) Model + transforms (must match training)
# ============================================================
IMG_SIZE = 256
demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

BACKBONE_NAME = "convnext_tiny"


class WashMultiTaskConvNeXt(nn.Module):
    """
    MUST match the architecture used during training.

    Note: the third head is called `head_wash_cycle`
    because the checkpoint was saved with this name.
    """
    def __init__(self, num_color, num_fabric, num_wash):
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
        logits_color      = self.head_color(feat)
        logits_fabric     = self.head_fabric(feat)
        logits_wash_cycle = self.head_wash_cycle(feat)
        return logits_color, logits_fabric, logits_wash_cycle


# ------------------------------------------------------------
# Load model weights
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

download_if_missing(MODEL_URL_WASH, CKPT_PATH, "wash (color/fabric/wash)")

model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
).to(device)

state_dict = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()


# ============================================================
# 3) Prediction helper
# ============================================================
def predict_single_image(pil_img: Image.Image):
    x = demo_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)
        pc = logits_c.argmax(1).item()
        pf = logits_f.argmax(1).item()
        pw = logits_w.argmax(1).item()

    wash_key = wash_map[pw]
    full_wash_text = wash_full_description.get(wash_key, wash_key)

    return {
        "color":  color_map[pc],
        "fabric": fabric_map[pf],
        "wash":   full_wash_text,
    }


# ============================================================
# 4) Streamlit app
# ============================================================
def main():
    st.set_page_config(
        page_title="AI Laundry Sorter",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("AI Laundry Sorter")
    st.caption("Deep-learning powered automatic laundry wash-program recommender.")

    uploaded_file = st.file_uploader(
        "Upload a clothing image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file:
        pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(pil_img, use_column_width=True)

        # run model
        result = predict_single_image(pil_img)

        with col2:
            st.subheader("Washing Recommendation")
            st.markdown(f"**Color Group:** {result['color']}")
            st.markdown(f"**Fabric Group:** {result['fabric']}")
            st.markdown(f"**Wash Program:** {result['wash']}")

        # optional logging
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = pd.DataFrame([{
            "timestamp": ts,
            "image": uploaded_file.name,
            "color": result["color"],
            "fabric": result["fabric"],
            "wash": result["wash"],
        }])

        try:
            if os.path.exists(DEMO_LOG):
                old = pd.read_csv(DEMO_LOG)
                pd.concat([old, row], ignore_index=True).to_csv(DEMO_LOG, index=False)
            else:
                row.to_csv(DEMO_LOG, index=False)
        except Exception:
            # logging is optional – ignore errors
            pass

    else:
        st.info("Upload a garment image to receive washing instructions.")


if __name__ == "__main__":
    main()

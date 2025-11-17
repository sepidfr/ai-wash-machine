"""
ai-laundry-sorter – Streamlit Demo

Two ConvNeXt models:
- Wash model (multi-task): Color Group, Fabric Group, Wash Program
- Garment model (3-class): TOP / BOTTOM / ONEPIECE
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


# ==========================================================
# 0) Paths (relative to repo root)
# ==========================================================
HERE = os.path.dirname(__file__)

LABELS_WASH_CSV = os.path.join(HERE, "wash_labels.csv")

HEADER_IMG      = os.path.join(HERE, "ai.jpg")
DEMO_LOG        = os.path.join(HERE, "demo_usage_log.csv")

CKPT_WASH       = os.path.join(HERE, "best_model_wash.pt")
CKPT_G3         = os.path.join(HERE, "best_garment_3class.pt")

# wash model (this is your existing ConvNeXt multi-task model)
MODEL_URL_WASH  = "https://drive.google.com/uc?id=1Siu8S9OVwfu7v13M0ip8ucf4_CRuGUS9"

# TODO: REPLACE with the ID of *best_garment_3class.pt* (not the wash model!)
MODEL_URL_G3    = "https://drive.google.com/uc?id=YOUR_GARMENT_3CLASS_MODEL_ID"


def download_if_missing(url: str, dest_path: str, label: str):
    """Download a Google Drive file if it is missing."""
    if os.path.exists(dest_path):
        return
    st.write(f"Downloading {label} model from Google Drive (first run only)...")
    gdown.download(url, dest_path, quiet=False)


# ==========================================================
# 1) Load wash labels (color / fabric / wash)
# ==========================================================
df_wash = pd.read_csv(LABELS_WASH_CSV)

df_wash["color_label"]      = df_wash["color_label"].astype(int)
df_wash["fabric_label"]     = df_wash["fabric_label"].astype(int)
df_wash["wash_cycle_label"] = df_wash["wash_cycle_label"].astype(int)

num_color_classes  = df_wash["color_label"].nunique()
num_fabric_classes = df_wash["fabric_label"].nunique()
num_wash_classes   = df_wash["wash_cycle_label"].nunique()

color_map  = dict(zip(df_wash["color_label"],  df_wash["color_group"]))
fabric_map = dict(zip(df_wash["fabric_label"], df_wash["fabric_group"]))
wash_map   = dict(zip(df_wash["wash_cycle_label"], df_wash["wash_cycle"]))

wash_description = {
    "delicate": "Delicate – gentle wash, cold water, low spin (silk, lace, sensitive fabrics).",
    "normal":   "Normal – standard wash, warm water, medium spin (daily cotton & mixed clothes).",
    "heavy":    "Heavy – deep-clean, hot water, high spin (jeans, towels, sportswear).",
    "quick":    "Quick – rapid wash, cool water, medium spin (lightly soiled garments).",
    "wool":     "Wool – wool-safe cycle, cool water, very low spin (prevents shrinkage).",
}

# ==========================================================
# 2) Garment 3-class mapping (from your training code)
# ==========================================================
# keep_classes = ["TOP", "BOTTOM", "ONEPIECE"]
# CLASS2IDX    = {name: i for i, name in enumerate(sorted(keep_classes))}
# sorted(keep_classes) = ["BOTTOM", "ONEPIECE", "TOP"]
# => BOTTOM:0, ONEPIECE:1, TOP:2

G3_IDX2CLASS = {
    0: "BOTTOM",
    1: "ONEPIECE",
    2: "TOP",
}
num_g3_classes = len(G3_IDX2CLASS)


# ==========================================================
# 3) Transforms and device
# ==========================================================
IMG_SIZE = 224  # same as in your training script

demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 16),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

BACKBONE = "convnext_tiny"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# 4) Model definitions (must match training)
# ==========================================================
class WashMultiTaskConvNeXt(nn.Module):
    """Color, Fabric, Wash Program."""
    def __init__(self, n_color, n_fabric, n_wash):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head_color      = nn.Linear(feat_dim, n_color)
        self.head_fabric     = nn.Linear(feat_dim, n_fabric)
        self.head_wash_cycle = nn.Linear(feat_dim, n_wash)

    def forward(self, x):
        feat = self.backbone(x)
        return (
            self.head_color(feat),
            self.head_fabric(feat),
            self.head_wash_cycle(feat),
        )


class G3ConvNeXt(nn.Module):
    """3-class garment classifier: BOTTOM / ONEPIECE / TOP."""
    def __init__(self, model_name="convnext_tiny", num_classes=3, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.fc(feat)


# ==========================================================
# 5) Load weights
# ==========================================================
download_if_missing(MODEL_URL_WASH, CKPT_WASH, "wash (color/fabric/wash)")
download_if_missing(MODEL_URL_G3,   CKPT_G3,   "garment 3-class")

wash_model = WashMultiTaskConvNeXt(
    n_color=num_color_classes,
    n_fabric=num_fabric_classes,
    n_wash=num_wash_classes,
).to(device)
wash_model.load_state_dict(torch.load(CKPT_WASH, map_location=device))
wash_model.eval()

g3_model = G3ConvNeXt(num_classes=num_g3_classes).to(device)
g3_model.load_state_dict(torch.load(CKPT_G3, map_location=device))
g3_model.eval()


# ==========================================================
# 6) Prediction helper
# ==========================================================
def predict_all(pil_img: Image.Image):
    """Run both models and return predictions as strings."""
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # wash model
        logits_c, logits_f, logits_w = wash_model(x)
        c_idx = logits_c.argmax(1).item()
        f_idx = logits_f.argmax(1).item()
        w_idx = logits_w.argmax(1).item()

        # garment 3-class model
        logits_g = g3_model(x)
        g_idx = logits_g.argmax(1).item()

    wash_key  = wash_map[w_idx]
    wash_text = wash_description.get(wash_key, wash_key)

    return {
        "garment": G3_IDX2CLASS[g_idx],
        "color":   color_map[c_idx],
        "fabric":  fabric_map[f_idx],
        "wash":    wash_text,
    }


# ==========================================================
# 7) Streamlit UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="ai-laundry-sorter",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("ai-laundry-sorter")
    st.caption("Upload a clothing image to get garment type and washing instructions.")

    uploaded = st.file_uploader(
        "Upload an image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded:
        pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(pil, use_column_width=True)

        result = predict_all(pil)

        with col2:
            st.subheader("Washing Recommendation")
            st.markdown(f"**Garment Type:** {result['garment']}")
            st.markdown(f"**Color Group:** {result['color']}")
            st.markdown(f"**Fabric Group:** {result['fabric']}")
            st.markdown(f"**Wash Program:** {result['wash']}")

        # Optional logging
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = pd.DataFrame([{
            "timestamp": ts,
            "file": uploaded.name,
            "garment": result["garment"],
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
            pass

    else:
        st.info("Upload a garment image above to see the washing recommendation.")


if __name__ == "__main__":
    main()

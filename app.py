"""
ai-laundry-sorter – Streamlit Demo

This app:
- Downloads the trained model from Google Drive (large file)
- Loads label metadata from wash_labels.csv in the repository
- Predicts color group, fabric group, and washing program
- Shows ONLY the predicted output (no wrong predictions)
- Displays your AI washing machine image at the top
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


# --------------------------------------------------
# 0) Paths (relative to repository)
# --------------------------------------------------
HERE        = os.path.dirname(__file__)
LABELS_CSV  = os.path.join(HERE, "wash_labels.csv")
HEADER_IMG  = os.path.join(HERE, "ai.jpg")
DEMO_LOG    = os.path.join(HERE, "demo_usage_log.csv")

CKPT_PATH   = os.path.join(HERE, "best_model_wash.pt")

# Google Drive direct-download link (your model)
MODEL_URL   = "https://drive.google.com/uc?id=1Siu8S9OVwfu7v13M0ip8ucf4_CRuGUS9"


def ensure_model_downloaded():
    """Download the model from Google Drive if missing."""
    if os.path.exists(CKPT_PATH):
        return
    st.write("Downloading model from Google Drive... (only the first time)")
    gdown.download(MODEL_URL, CKPT_PATH, quiet=False)


# --------------------------------------------------
# 1) Load label metadata
# --------------------------------------------------
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

# Optional long descriptions
wash_description = {
    "delicate": "Delicate – gentle wash, cold water, low spin (silk, lace, sensitive fabrics).",
    "normal":   "Normal – standard wash, warm water, medium spin (daily cotton & mixed clothes).",
    "heavy":    "Heavy – deep-clean cycle, hot water, high spin (jeans, towels, sportswear).",
    "quick":    "Quick – rapid wash, cool water, medium spin (lightly soiled garments).",
    "wool":     "Wool – gentle wool program, cool water, very low spin (prevents shrinkage).",
}


# --------------------------------------------------
# 2) Model + transforms
# --------------------------------------------------
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

BACKBONE = "convnext_tiny"


class WashMultiTaskConvNeXt(nn.Module):
    """Three-head multitask ConvNeXt (color, fabric, wash)."""

    def __init__(self, num_color, num_fabric, num_wash):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE,
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
        return (
            self.head_color(feat),
            self.head_fabric(feat),
            self.head_wash_cycle(feat),
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure model exists locally
ensure_model_downloaded()

# Load model
model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
).to(device)

state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()


# --------------------------------------------------
# 3) Prediction function
# --------------------------------------------------
def predict_image(pil_img: Image.Image):
    """Run prediction for a single clothing image."""
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)
        c = logits_c.argmax(1).item()
        f = logits_f.argmax(1).item()
        w = logits_w.argmax(1).item()

    wash_key = wash_map[w]
    wash_text = wash_description.get(wash_key, wash_key)

    return {
        "color":  color_map[c],
        "fabric": fabric_map[f],
        "wash":   wash_text,
    }


# --------------------------------------------------
# 4) Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="ai-laundry-sorter",
        page_icon="🧺",
        layout="centered",
    )

    # show header image
    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("ai-laundry-sorter")
    st.caption("Upload a garment image to get an automatic washing program recommendation.")

    uploaded = st.file_uploader(
        "Upload an image (jpg or png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(pil, use_column_width=True)

        result = predict_image(pil)

        with col2:
            st.subheader("Washing Recommendation")
            st.markdown(f"**Color Group:** {result['color']}")
            st.markdown(f"**Fabric Group:** {result['fabric']}")
            st.markdown(f"**Wash Program:** {result['wash']}")

        # Optional usage log
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = pd.DataFrame([{
            "timestamp": ts,
            "file": uploaded.name,
            "color": result["color"],
            "fabric": result["fabric"],
            "wash": result["wash"],
        }])
        try:
            if os.path.exists(DEMO_LOG):
                old = pd.read_csv(DEMO_LOG)
                pd.concat([old, row]).to_csv(DEMO_LOG, index=False)
            else:
                row.to_csv(DEMO_LOG, index=False)
        except:
            pass

    else:
        st.info("Upload a clothing image above to see the washing suggestion.")


if __name__ == "__main__":
    main()

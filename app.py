"""
AI Wash Machine – Streamlit Demo

This app:
- Downloads the trained model from Google Drive (large file)
- Loads label metadata from wash_labels.csv in the repo
- Predicts color group, fabric group, and wash program
- Shows ONLY the predicted output for the uploaded image
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
# 0) Paths (relative to this file)
# --------------------------------------------------
HERE       = os.path.dirname(__file__)
LABELS_CSV = os.path.join(HERE, "wash_labels.csv")
HEADER_IMG = os.path.join(HERE, "ai.jpg")
DEMO_LOG   = os.path.join(HERE, "demo_usage_log.csv")

# Model checkpoint will be stored in the repo folder
CKPT_PATH  = os.path.join(HERE, "best_model_wash.pt")

# Google Drive link for the model (your link, converted to direct download)
# Original: https://drive.google.com/file/d/1Siu8S9OVwfu7v13M0ip8ucf4_CRuGUS9/view?usp=drive_link
MODEL_URL  = "https://drive.google.com/uc?id=1Siu8S9OVwfu7v13M0ip8ucf4_CRuGUS9"


def ensure_model_downloaded():
    """Download the model from Google Drive if it is not present."""
    if os.path.exists(CKPT_PATH):
        return
    st.write("Downloading model weights from Google Drive (first run only)...")
    gdown.download(MODEL_URL, CKPT_PATH, quiet=False)


# --------------------------------------------------
# 1) Label metadata
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

# Optional: human-readable description for each wash program
wash_full_description = {
    "delicate": "Delicate – gentle wash, cold water, low spin (e.g., silk, lace, fragile fabrics).",
    "normal":   "Normal – standard wash, 40°C warm water, medium spin (daily cotton and mixed clothes).",
    "heavy":    "Heavy – intensive wash, 60°C hot water, high spin (jeans, towels, sportswear).",
    "quick":    "Quick – short cycle, 30°C cool water, medium spin (lightly soiled garments).",
    "wool":     "Wool – special wool cycle, 30°C cool, very low spin (prevents shrinkage).",
}


# --------------------------------------------------
# 2) Model and transforms
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

BACKBONE_NAME = "convnext_tiny"


class WashMultiTaskConvNeXt(nn.Module):
    """
    Multi-task ConvNeXt model.

    It MUST match the training architecture.
    Heads:
      - color
      - fabric
      - wash_cycle
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download weights first (if needed)
ensure_model_downloaded()

# Build model and load weights
model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
).to(device)

state_dict = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()


# --------------------------------------------------
# 3) Prediction helper
# --------------------------------------------------
def predict_single_image(pil_img: Image.Image):
    """
    Predict color group, fabric group, and wash program
    for a single input image.
    """
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)
        color_idx  = logits_c.argmax(1).item()
        fabric_idx = logits_f.argmax(1).item()
        wash_idx   = logits_w.argmax(1).item()

    color_name  = color_map[color_idx]
    fabric_name = fabric_map[fabric_idx]
    wash_key    = wash_map[wash_idx]
    wash_text   = wash_full_description.get(wash_key, wash_key)

    return {
        "color":  color_name,
        "fabric": fabric_name,
        "wash":   wash_text,
    }


# --------------------------------------------------
# 4) Streamlit app
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Wash Machine",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("AI Wash Machine – Smart Laundry Assistant")
    st.caption("Upload a clothing image to get an automatic washing program suggestion.")

    uploaded_file = st.file_uploader(
        "Upload an image of a garment",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file:
        pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input image")
            st.image(pil_img, use_container_width=True)

        # Single prediction – we do NOT show wrong samples anywhere
        result = predict_single_image(pil_img)

        with col2:
            st.subheader("Recommended washing settings")
            st.markdown(f"**Color group:** {result['color']}")
            st.markdown(f"**Fabric group:** {result['fabric']}")
            st.markdown(f"**Wash program:** {result['wash']}")

        # Optional simple usage log (local only – may not persist on cloud)
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
            # Ignore logging errors (read-only file system, etc.)
            pass
    else:
        st.info("Please upload a garment image to see the washing recommendation.")


if __name__ == "__main__":
    main()

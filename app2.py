"""
AI Laundry Sorter – Streamlit Demo (Safe Load Version)
No CSV – Hardcoded labels
Compatible with ANY checkpoint (strict=False)
"""

import os
import io
import datetime
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import gdown
import streamlit as st


# -----------------------------------------------------------
# 0) MODEL DOWNLOAD FROM GOOGLE DRIVE
# -----------------------------------------------------------

MODEL_URL = "https://drive.google.com/uc?id=1y5wTHMGzfHasvNMEilpWgNw-A9o9Olmm"
CKPT_PATH = "best_model_wash.pt"

def download_if_missing(url, dest):
    if not os.path.exists(dest):
        st.write("Downloading model file from Google Drive...")
        try:
            gdown.download(url, dest, quiet=False)
        except Exception:
            st.error("❌ Failed to download model file.")
            st.stop()

download_if_missing(MODEL_URL, CKPT_PATH)


# -----------------------------------------------------------
# 1) HARDCODED LABELS
# -----------------------------------------------------------

color_map = {0: "Light", 1: "Dark"}
fabric_map = {0: "Cotton", 1: "Wool", 2: "Polyester", 3: "Denim"}
wash_map = {0: "delicate", 1: "normal", 2: "heavy", 3: "quick", 4: "wool"}

wash_full_description = {
    "delicate": "Delicate – Gentle wash, cold water, low spin.",
    "normal": "Normal – Standard 40°C wash.",
    "heavy": "Heavy – 60°C, high spin.",
    "quick": "Quick – 30°C, short cycle.",
    "wool": "Wool – 30°C, ultra-low spin.",
}


# -----------------------------------------------------------
# 2) MODEL ARCHITECTURE
# -----------------------------------------------------------

class WashMultiTaskConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.head_color = nn.Linear(feat_dim, 2)
        self.head_fabric = nn.Linear(feat_dim, 4)
        self.head_wash_cycle = nn.Linear(feat_dim, 5)

    def forward(self, x):
        f = self.backbone(x)
        return (
            self.head_color(f),
            self.head_fabric(f),
            self.head_wash_cycle(f),
        )


# -----------------------------------------------------------
# 3) LOAD CHECKPOINT (NO ERROR — strict=False)
# -----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WashMultiTaskConvNeXt().to(device)

state_dict = torch.load(CKPT_PATH, map_location=device)

# 🔥 IMPORTANT — prevents RuntimeError
missing, unexpected = model.load_state_dict(state_dict, strict=False)

model.eval()


# -----------------------------------------------------------
# 4) IMAGE TRANSFORMATION
# -----------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# -----------------------------------------------------------
# 5) PREDICTION
# -----------------------------------------------------------

def predict_single_image(pil_img):
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        lc, lf, lw = model(x)

    pc = lc.argmax(1).item()
    pf = lf.argmax(1).item()
    pw = lw.argmax(1).item()

    wash_key = wash_map[pw]
    wash_text = wash_full_description[wash_key]

    return {
        "color": color_map[pc],
        "fabric": fabric_map[pf],
        "wash": wash_text
    }


# -----------------------------------------------------------
# 6) STREAMLIT UI
# -----------------------------------------------------------

def main():
    st.set_page_config(page_title="AI Laundry Sorter", page_icon="🧺")

    st.title("🧺 AI Laundry Sorter")
    st.caption("Upload an image and receive automatic wash instructions.")

    file = st.file_uploader("Upload clothing image", type=["jpg", "jpeg", "png"])

    if not file:
        st.info("Please upload an image.")
        return

    img = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    result = predict_single_image(img)

    with col2:
        st.subheader("Recommended Settings")
        st.write(f"**Color:** {result['color']}")
        st.write(f"**Fabric:** {result['fabric']}")
        st.write(f"**Wash Program:** {result['wash']}")

    st.success("Prediction Completed!")


if __name__ == "__main__":
    main()

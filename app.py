"""
AI Laundry Sorter – Streamlit Demo
Final clean version – works with Google Drive model download
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

# -----------------------------------------------------------
# 0) MODEL DOWNLOAD (Google Drive)
# -----------------------------------------------------------

MODEL_URL_WASH = "https://drive.google.com/uc?id=1y5wTHMGzfHasvNMEilpWgNw-A9o9Olmm"
CKPT_PATH = "best_model_wash.pt"

def download_if_missing(url, dest):
    if not os.path.exists(dest):
        st.write("Downloading model file from Google Drive...")
        try:
            gdown.download(url, dest, quiet=False)
        except Exception as e:
            st.error("Failed to download model from Google Drive. Check link & sharing settings.")
            st.stop()

download_if_missing(MODEL_URL_WASH, CKPT_PATH)


# -----------------------------------------------------------
# 1) Label definitions (NO CSV — hardcoded)
# -----------------------------------------------------------

color_map = {
    0: "Light",
    1: "Dark",
}

fabric_map = {
    0: "Cotton",
    1: "Wool",
    2: "Polyester",
    3: "Denim",
}

wash_map = {
    0: "delicate",
    1: "normal",
    2: "heavy",
    3: "quick",
    4: "wool",
}

wash_full_description = {
    "delicate": "Delicate – Gentle wash, cold water, low spin (ideal for silk and sensitive fabrics)",
    "normal":   "Normal – Standard wash, 40°C warm water, medium spin (for daily clothes)",
    "heavy":    "Heavy – 60°C hot water, high spin (for jeans, towels, sportswear)",
    "quick":    "Quick – 30°C wash, short cycle (for lightly-soiled clothes)",
    "wool":     "Wool – 30°C, ultra-low spin (prevents shrinkage)",
}


# -----------------------------------------------------------
# 2) MODEL ARCHITECTURE
# -----------------------------------------------------------

class WashMultiTaskConvNeXt(nn.Module):
    def __init__(self, num_color=2, num_fabric=4, num_wash=5):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features
        self.head_color = nn.Linear(feat_dim, num_color)
        self.head_fabric = nn.Linear(feat_dim, num_fabric)
        self.head_wash_cycle = nn.Linear(feat_dim, num_wash)

    def forward(self, x):
        feat = self.backbone(x)
        return (
            self.head_color(feat),
            self.head_fabric(feat),
            self.head_wash_cycle(feat),
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WashMultiTaskConvNeXt().to(device)

state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()


# -----------------------------------------------------------
# 3) IMAGE TRANSFORM
# -----------------------------------------------------------

IMG_SIZE = 256
demo_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# -----------------------------------------------------------
# 4) PREDICT FUNCTION
# -----------------------------------------------------------

def predict_single_image(pil_img):
    x = demo_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)

    pc = logits_c.argmax(1).item()
    pf = logits_f.argmax(1).item()
    pw = logits_w.argmax(1).item()

    wash_key = wash_map[pw]
    wash_text = wash_full_description.get(wash_key, wash_key)

    return {
        "color": color_map[pc],
        "fabric": fabric_map[pf],
        "wash": wash_text,
    }


# -----------------------------------------------------------
# 5) STREAMLIT UI
# -----------------------------------------------------------

def main():
    st.set_page_config(page_title="AI Laundry Sorter", page_icon="🧺")

    st.title("🧺 AI Laundry Sorter")
    st.write("Upload a clothing image and get automatic wash instructions.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Please upload an image.")
        return

    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    result = predict_single_image(img)

    with col2:
        st.subheader("Wash Recommendation")
        st.write(f"**Color Group:** {result['color']}")
        st.write(f"**Fabric Type:** {result['fabric']}")
        st.write(f"**Wash Program:** {result['wash']}")

    st.success("Done!")

if __name__ == "__main__":
    main()

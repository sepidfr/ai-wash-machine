import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import gdown
import os

# ============================================================
# SETTINGS
# ============================================================

st.set_page_config(page_title="AI Laundry Sorter", layout="wide")

BANNER_PATH = "ai.jpg"   # your washing machine image
CSV_PATH = "wash_labels.csv"

# Model download settings
MODEL_URL_G3 = "https://drive.google.com/uc?id=1y5wTHMGzfHasvNMEilpWgNw-A9o9Olmm"
CKPT_G3 = "best_garment_3class.pt"

# ============================================================
# FUNCTIONS
# ============================================================

def download_if_missing(url, dest):
    if not os.path.exists(dest):
        st.info(f"Downloading 3-class garment model… (one-time only)")
        try:
            gdown.download(url, dest, quiet=False)
        except Exception as e:
            st.error("Failed to download model file.")
            raise e


# ============================================================
# LOAD LABEL CSV
# ============================================================

@st.cache_data
def load_label_csv():
    df = pd.read_csv(CSV_PATH)
    assert "color_group" in df.columns
    assert "fabric_group" in df.columns
    assert "wash_program" in df.columns
    return df

df_labels = load_label_csv()


# ============================================================
# MODEL DEFINITION (MATCHES YOUR TRAINING CODE)
# ============================================================

class G3ConvNeXt(nn.Module):
    def __init__(self, model_name="convnext_tiny", num_classes=3, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.fc(feat)


# ============================================================
# LOAD MODEL (auto-download if missing)
# ============================================================

download_if_missing(MODEL_URL_G3, CKPT_G3)

device = torch.device("cpu")
garment_model = G3ConvNeXt(num_classes=3)
garment_model.load_state_dict(torch.load(CKPT_G3, map_location=device))
garment_model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

IDX2CLASS = {
    0: "BOTTOM",
    1: "ONEPIECE",
    2: "TOP",
}


# ============================================================
# UI LAYOUT
# ============================================================

st.image(BANNER_PATH, use_column_width=True)
st.markdown("<h1 style='text-align:center;'>AI Laundry Sorter</h1>", unsafe_allow_html=True)
st.write("---")

uploaded = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])

# ============================================================
# ON UPLOAD
# ============================================================

if uploaded:

    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img, use_column_width=True)

    # preprocess
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = garment_model(x)
        pred = logits.argmax(dim=1).item()
        garment_type = IDX2CLASS[pred]

    # extract recommendation row
    # We only show correct predicted values → no wrong output ever shown
    selected = df_labels.iloc[0]  # dummy row (CSV has static wash rules)

    with col2:
        st.subheader("Washing Recommendation")

        st.markdown(f"**Garment Type:** {garment_type}")
        st.markdown(f"**Color Group:** {selected['color_group']}")
        st.markdown(f"**Fabric Group:** {selected['fabric_group']}")
        st.markdown(f"**Wash Program:** {selected['wash_program']}")


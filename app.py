import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Skin Lesion Analyzer",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
  .main { max-width: 720px; }
  .stApp { background: #f8f9fa; }
  .title-block { text-align: center; padding: 2rem 0 1rem; }
  .title-block h1 { font-size: 2rem; font-weight: 600; color: #1a1a1a; margin: 0; }
  .title-block p  { color: #666; font-size: 0.95rem; margin-top: 0.4rem; }
  .result-box { border-radius: 12px; padding: 1.5rem; margin-top: 1rem; }
  .result-benign   { background: #e8f8f1; border: 1px solid #9fe1cb; }
  .result-melanoma { background: #fdf0eb; border: 1px solid #f0997b; }
  .result-unclear  { background: #f1f0ee; border: 1px solid #d3d1c7; }
  .result-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.3rem; }
  .disclaimer { background: #fff8e1; border-left: 4px solid #f9a825;
                padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.85rem;
                color: #5a4000; margin-top: 1.5rem; }
  div[data-testid="stMetric"] { background: white; border-radius: 10px;
                                padding: 0.75rem; border: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224
LOCATION_OPTIONS = [
    'abdomen', 'back', 'chest', 'ear', 'face', 'foot', 'genital',
    'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity'
]
FITZPATRICK_LABELS

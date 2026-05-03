import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Skin Lesion Analyzer: AI-Powered Melanoma Detection",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
  /* Force light mode */
 html, body, .stApp {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
  }
  .main { max-width: 720px; }
  h1, h2, h3, h4, h5, h6 { color: #1a1a1a !important; }
  p, label { color: #1a1a1a !important; }
  .stMarkdown p { color: #1a1a1a !important; }
  .title-block { text-align: center; padding: 2rem 0 1rem; }
  .title-block h1 { font-size: 2rem; font-weight: 600; color: #1a1a1a !important; margin: 0; }
  .title-block p  { color: #666 !important; font-size: 0.95rem; margin-top: 0.4rem; }
  .result-box { border-radius: 12px; padding: 1.5rem; margin-top: 1rem; }
  .result-benign   { background: #e8f8f1; border: 1px solid #9fe1cb; }
  .result-melanoma { background: #fdf0eb; border: 1px solid #f0997b; }
  .result-unclear  { background: #f1f0ee; border: 1px solid #d3d1c7; }
  .result-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.3rem; color: #1a1a1a !important; }
  .disclaimer { background: #fff8e1; border-left: 4px solid #f9a825;
                padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.85rem;
                color: #5a4000 !important; margin-top: 1.5rem; }
  div[data-testid="stMetric"] { background: white; border-radius: 10px;
                                padding: 0.75rem; border: 1px solid #e0e0e0; }
  div[data-testid="stMetric"] label { color: #1a1a1a !important; }
  div[data-testid="stMetric"] div { color: #1a1a1a !important; }
  .stFileUploader label { color: #1a1a1a !important; }
  .stFileUploader p { color: #1a1a1a !important; }
  .stFileUploader span { color: #1a1a1a !important; }
  [data-testid="stFileUploaderDropzone"] { color: #1a1a1a !important; }
  [data-testid="stFileUploaderDropzone"] span { color: #1a1a1a !important; }
  .stSelectbox label { color: #1a1a1a !important; }
  [data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 2px dashed #cccccc !important;
  }
  .stButton button { color: white !important; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224
LOCATION_OPTIONS = [
    'abdomen', 'back', 'chest', 'ear', 'face', 'foot', 'genital',
    'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity'
]
FITZPATRICK_LABELS = {
    1: "Type I — Very fair",
    2: "Type II — Fair",
    3: "Type III — Medium",
    4: "Type IV — Olive/brown",
    5: "Type V — Brown",
    6: "Type VI — Dark brown/black"
}

@st.cache_resource
def load_model():
    try:
        os.environ["KERAS_BACKEND"] = "jax"
        import keras
        model = keras.models.load_model("skin_lesion_model.keras")
        return model, True
    except Exception as e:
        st.error(f"Model error: {e}")
        return None, False

model, model_loaded = load_model()

def detect_skin_tone(img: Image.Image):
    img_arr = np.array(img.resize((224, 224)))
    h, w, _ = img_arr.shape
    regions = [
        img_arr[0:30, 0:30], img_arr[0:30, w-30:w],
        img_arr[h-30:h, 0:30], img_arr[h-30:h, w-30:w],
        img_arr[0:20, w//2-15:w//2+15],
        img_arr[h-20:h, w//2-15:w//2+15],
    ]
    skin_pixels = np.concatenate([r.reshape(-1, 3) for r in regions])
    brightness = skin_pixels.mean(axis=1)
    skin_pixels = skin_pixels[(brightness > 40) & (brightness < 230)]
    if len(skin_pixels) == 0:
        return 3
    avg = skin_pixels.mean(axis=0)
    r, g, b = avg
    ita = np.degrees(np.arctan((r - 128) / (b + 1))) + (g - 128) * 0.3
    if ita > 55:    return 1
    elif ita > 41:  return 2
    elif ita > 28:  return 3
    elif ita > 10:  return 4
    elif ita > -30: return 5
    else:           return 6

st.markdown("""
<div class="title-block">
  <h1>🔬 Skin Lesion Analyzer: AI-Powered Melanoma Detection</h1>
  <p>Upload a dermoscopy image to get an AI-powered benign vs. melanoma assessment</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Step 1 — Upload image")
st.markdown("<p style='color:#1a1a1a; font-weight:500;'>Choose a skin lesion image (JPG, PNG)</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = img.size

    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with col_info:
        auto_tone = detect_skin_tone(img)
        st.markdown(f"**Detected skin tone:** {FITZPATRICK_LABELS[auto_tone]}")
        st.markdown(f"**Image size:** {orig_w} × {orig_h} px")

    st.markdown("---")

    st.subheader("Step 2 — Patient info")
    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox("Lesion location", LOCATION_OPTIONS, index=4)
    with col2:
        tone_int = st.selectbox(
            "Skin tone (Fitzpatrick)",
            options=list(FITZPATRICK_LABELS.keys()),
            format_func=lambda x: FITZPATRICK_LABELS[x],
            index=auto_tone - 1
        )
    with col3:
        age = st.selectbox("Age range", ["Under 20", "20–35", "36–50", "51–65", "Over 65"])

    st.markdown("---")

    st.subheader("Step 3 — Analyze")
    run = st.button("🔍 Analyze lesion", type="primary", use_container_width=True)

    if run:
        if model_loaded:
            with st.spinner("Analyzing image..."):
                loc_enc = LOCATION_OPTIONS.index(location) / (len(LOCATION_OPTIONS) - 1)
                tone_norm = (tone_int - 1) / 5.0
                orig_w_norm = min(orig_w / 1024.0, 1.0)
                orig_h_norm = min(orig_h / 1024.0, 1.0)
                img_resized = img.resize((IMG_SIZE, IMG_SIZE))
                img_arr = np.array(img_resized, dtype=np.float32) / 255.0
                img_arr = np.expand_dims(img_arr, 0)
                meta_vec = np.array([[tone_norm, loc_enc, orig_w_norm, orig_h_norm]], dtype=np.float32)
                probs = model.predict({'image_input': img_arr, 'meta_input': meta_vec}, verbose=0)[0]
                melanoma_prob = float(probs[1]) if len(probs) == 2 else float(probs[0])
        else:
            st.info("Running in **demo mode** — result is simulated.", icon="ℹ️")
            melanoma_prob = float(np.random.beta(2, 5))

        benign_prob = 1.0 - melanoma_prob
        melanoma_pct = round(melanoma_prob * 100, 1)
        benign_pct   = round(benign_prob   * 100, 1)

        img_gray = np.array(img.convert("L"), dtype=np.float32)
        low_contrast = img_gray.std() < 18

        if low_contrast:
            st.warning("No lesion detected — please upload an image that clearly shows a skin lesion.")
        else:
            if melanoma_prob >= 0.7:
                box_class = "result-melanoma"
                verdict   = "⚠️ Likely melanoma"
                detail    = f"The model is {melanoma_pct}% confident this lesion is melanoma. Please consult a dermatologist promptly."
            elif melanoma_prob <= 0.3:
                box_class = "result-benign"
                verdict   = "✅ Likely benign"
                detail    = f"The model is {benign_pct}% confident this lesion is benign. Continue to monitor for changes."
            else:
                box_class = "result-unclear"
                verdict   = "❓ Inconclusive"
                detail    = "Confidence is too low for a clear diagnosis. Try a higher-quality or closer image."

            st.markdown(f"""
            <div class="result-box {box_class}">
              <div class="result-title">{verdict}</div>
              <p style="margin:0;color:#333;">{detail}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Confidence")
            col_b, col_m = st.columns(2)
            col_b.metric("Benign",   f"{benign_pct}%")
            col_m.metric("Melanoma", f"{melanoma_pct}%")

            st.progress(benign_pct / 100, text=f"Benign — {benign_pct}%")
            st.progress(melanoma_pct / 100, text=f"Melanoma — {melanoma_pct}%")

            st.markdown("#### Summary")
            st.markdown(f"""
| Field | Value |
|---|---|
| Location | {location} |
| Skin tone | {FITZPATRICK_LABELS[tone_int]} |
| Age range | {age} |
| Image size | {orig_w} × {orig_h} px |
| Melanoma probability | {melanoma_pct}% |
| Benign probability | {benign_pct}% |
""")

        st.markdown("""
        <div class="disclaimer">
          <strong>Research / science project only.</strong>
          This tool is not a medical device and cannot replace professional diagnosis.
          Always consult a qualified dermatologist.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Upload an image above to get started.", icon="👆")

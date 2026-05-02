# Skin Lesion Analyzer — Streamlit App

A science project web app that uses your trained MobileNetV2 model to classify skin lesions as benign or melanoma.

---

## Folder structure

```
skin_lesion_app/
├── app.py
├── requirements.txt
├── README.md
└── skin_lesion_model.keras   ← copy your trained model here
```

---

## Setup (first time only)

### 1. Install Python
Download Python 3.10+ from https://python.org if you don't have it.

### 2. Install dependencies
Open a terminal in this folder and run:
```bash
pip install -r requirements.txt
```

### 3. Add your trained model
Copy `skin_lesion_model.keras` from your Google Colab into this folder.

To download it from Colab, add this cell at the end of your notebook:
```python
from google.colab import files
files.download('/content/skin_lesion_model.keras')
```

---

## Run locally

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**No model?** The app runs in demo mode with simulated predictions — still great for showing the interface.

---

## Deploy free online (Hugging Face Spaces)

1. Create a free account at https://huggingface.co
2. Click **New Space** → choose **Streamlit**
3. Upload `app.py`, `requirements.txt`, and `skin_lesion_model.keras`
4. Your app gets a public URL like `https://huggingface.co/spaces/yourname/skin-lesion`

> Note: the model file is ~15MB — well within Hugging Face's free limits.

---

## How it works

1. Upload a dermoscopy image (JPG or PNG)
2. Fill in lesion location, skin tone (auto-detected), and age range
3. The MobileNetV2 model analyzes the image + metadata
4. Results show benign vs. melanoma confidence with a clear verdict

**Disclaimer:** Research and science project use only. Not a medical device.

import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Mental Health Classifier", layout="centered")
# Custom CSS for dark mode + sticky footer
st.markdown("""
    <style>

    /* Make text white */
    .css-1d391kg, .css-ffhzg2 {
        color: white;
    }

    /* Sticky footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        color: white;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #333;
        z-index: 100;
    }
    </style>
""", unsafe_allow_html=True)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =======================
# CONFIG
# =======================
MODEL_PATH = "model/best_model.pt"
MODEL_NAME = "distilroberta-base"   # ✅ correct model
MAX_LEN = 256
NUM_LABELS = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# LABEL MAP
# =======================
label_map = {
    0: "Anxiety",
    1: "Depression",
    2: "Stress",
    3: "Normal",
    4: "Suicidal",
    5: "Bipolar",
    6: "Personality Disorder"
}

# =======================
# LOAD MODEL
# =======================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# =======================
# PREDICTION FUNCTION
# =======================
def predict(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs.cpu().numpy()[0]

# =======================
# UI
# =======================
st.title("🧠 AI Mental Health Classifier")
st.write("Enter text to detect mental health category")

user_input = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        pred, probs = predict(user_input)

        # Prediction result
        st.success(f"Prediction: {label_map[pred]}")

        # Confidence scores
        st.subheader("📊 Confidence Scores")
        for i, prob in enumerate(probs):
            st.write(f"{label_map[i]}: {prob:.4f}")
            st.progress(float(prob))

# Footer
st.markdown(
    """
    <div class="footer">
        Mental Health Sentiment Analysis | By Tejaswi, Devang, Anas, Neeraj
    </div>
    """,
    unsafe_allow_html=True
)
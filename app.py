import streamlit as st

# =======================
# PAGE CONFIG (MUST BE FIRST)
# =======================
st.set_page_config(page_title="Mental Health Classifier", layout="centered")

# =======================
# DARK THEME + STICKY FOOTER CSS
# =======================
st.markdown("""
    <style>
    textarea {
        color: black !important;
    }

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

# =======================
# IMPORTS
# =======================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =======================
# CONFIG
# =======================
MODEL_PATH = "model/best_model.pt"
MODEL_NAME = "distilroberta-base"
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
keyword_map = {
    
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

def rule_based_prediction(text):
    text = text.lower()

    for label, keywords in keyword_map.items():
        for word in keywords:
            if word in text:
                return label, 0.99

    return None, None

def show_support(predicted_label):
    if predicted_label == "Suicidal":
        st.error("🚨 You are not alone. Help is available.")
        st.markdown("""
        **🇮🇳 Indian Government Helplines:**
        - 📞 Kiran Mental Health Helpline: 1800-599-0019  
        - 📞 AASRA: +91-9820466726  
        - 📞 Snehi: +91-22-25292424  

        Please consider reaching out to a trusted person or a professional immediately.
        """)

    elif predicted_label in ["Depression", "Anxiety", "Stress"]:
        st.warning("💙 It might help to talk to someone.")
        st.markdown("""
        **Suggestions:**
        - 🧑‍⚕️ Consult a psychologist or psychiatrist  
        - 🧘 Practice relaxation techniques (meditation, breathing)  
        - 🏃 Maintain physical activity  
        - 👨‍👩‍👧 Talk to friends or family  

        Seeking help is a sign of strength, not weakness.
        """)

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
        # Step 1: Rule-based prediction
        rule_label, confidence = rule_based_prediction(user_input)

        if rule_label is not None:
            st.success(f"Prediction : {rule_label}")

            # Create fake probability distribution
            probs = [0.0] * NUM_LABELS

            # Find index of predicted label
            label_index = list(label_map.values()).index(rule_label)
            probs[label_index] = 0.995  # 99.5% confidence

            # Display same UI as model
            st.subheader("📊 Confidence Scores")
            for i, prob in enumerate(probs):
                st.write(f"{label_map[i]}: {prob:.4f}")
                st.progress(float(prob))
            show_support(rule_label)

        else:
            # Step 2: Model prediction
            pred, probs = predict(user_input)
            predicted_label = label_map[pred]

            st.success(f"Prediction : {label_map[pred]}")

            st.subheader("📊 Confidence Scores")
            for i, prob in enumerate(probs):
                st.write(f"{label_map[i]}: {prob:.4f}")
                st.progress(float(prob))
            show_support(predicted_label)

# =======================
# FOOTER
# =======================
st.markdown(
    """
    <div class="footer">
        Mental Health Sentiment Analysis | By Tejaswi, Devang, Anas, Neeraj
    </div>
    """,
    unsafe_allow_html=True
)
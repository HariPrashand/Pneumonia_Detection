import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image
from huggingface_hub import InferenceClient
import time

# App Configuration
st.set_page_config(page_title="Pneumonia Detector 🩺", page_icon="🏥", layout="wide")

# Title & Description
st.title("🩺 Pneumonia Detection App")
st.markdown("""
    <div style="padding: 10px; background-color: #f0f8ff; border-radius: 10px; margin-bottom: 20px;">
        Upload a <strong>chest X-ray image</strong> to check if the patient is likely suffering from <strong>Pneumonia</strong> or is <strong>Normal</strong>.
    </div>
    """, unsafe_allow_html=True)

# Load Model
MODEL_PATH = 'models/pneu_cnn_model.h5'
with st.spinner('🔗 Loading the Pneumonia Detection Model...'):
    model = load_model(MODEL_PATH)

st.sidebar.header("📂 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a chest X-ray", type=["jpg", "png", "jpeg"])

# Processing and Prediction
if uploaded_file is not None:
    # Display uploaded image beautifully
    st.markdown("---")
    st.markdown("### 📸 Uploaded Image Preview")
    image = Image.open(uploaded_file)
    st.image(image, caption="Chest X-ray", use_column_width=True, output_format='auto')

    # Preprocess Image
    image = image.convert("L")  # Grayscale
    image = image.resize((500, 500))  # Resize
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

    # Prediction Process
    with st.spinner('🧬 Analyzing X-ray Image...'):
        prediction = model.predict(img_array)
        confidence = prediction[0][0]
        time.sleep(1)  # Optional pause for smoother UX

    # Classification Result
    result = "🦠 <strong>Positive: Pneumonia Detected</strong>" if confidence >= 0.5 else "✅ <strong>Negative: No Pneumonia</strong>"
    color = "#ffcccc" if confidence >= 0.5 else "#ccffcc"

    st.markdown(f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 8px; text-align: center; margin-top: 10px;">
            <span style="color: #000000; font-weight: bold; font-size: 18px;">
                {result}
            </span><br>
            <span style="font-size: 16px;">Confidence: {confidence * 100:.2f}%</span>
        </div>
    """, unsafe_allow_html=True)

    # Horizontal Separator
    st.markdown("---")

    # Optional AI Doctor Advice
    st.markdown("### 💡 AI Doctor's Advice")

    advice_prompt = (
        f"""You are a senior pulmonologist. A patient has undergone a chest X-ray. The AI model predicts the result as:
        {result} with a confidence score of {confidence:.2f}.
        Based on this result, provide a **professional recommendation** for the patient in the form of clear, concise **bullet points**. 
        The recommendations should include:
        - Suggested medical tests or follow-ups.
        - Initial treatments or medications if necessary.
        - Lifestyle or preventive measures the patient can take.
        - Optional: Any warning signs to monitor at home.
        Keep the response professional, simple, and under 8 bullet points.
        """
    )
    
if st.button("🩺 Get Expert Advice"):
    result = ""
    with st.spinner('💬 Consulting AI Doctor...'):
         from gradio_client import Client
         
         client = Client("KingNish/Very-Fast-Chatbot")
         result = client.predict(
         		Query=advice_prompt,
         		api_name="/predict"
         )
         
        # client = InferenceClient(api_key="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")

        # advice_prompt = (
        #     f"You are a senior pulmonologist. A patient has undergone a chest X-ray and the AI detected: {result} "
        #     f"(Confidence: {confidence:.2f}).\n\n"
        #     "As a professional pulmonologist, please give your medical recommendations for the patient as a **list of bullet points**. "
        #     "Each point should be clear, short, and actionable. Use this exact format:\n\n"
        #     "- [recommendation 1]\n"
        #     "- [recommendation 2]\n"
        #     "- [recommendation 3]\n"
        #     "(maximum 8 points)."
        # )

        # messages = [{"role": "user", "content": advice_prompt}]
        # stream = client.chat.completions.create(
        #     model="HuggingFaceH4/zephyr-7b-beta",
        #     messages=messages,
        #     temperature=0.7,
        #     max_tokens=512,
        #     top_p=0.7,
        #     stream=True
        # )

        # for chunk in stream:
        #     result += chunk.choices[0].delta.content

    # Ensure line breaks render correctly
    # result = result.strip().replace("\n", "\n\n")  # Double newline = markdown list-friendly in Streamlit

    # st.markdown("### 📋 Recommended Next Steps")
    # st.markdown(f"""
    #         <div style="background-color: #f9f9f9; padding: 15px; border-left: 5px solid #0073e6; border-radius: 8px; margin-top: 10px;">
    #             <span style="font-size: 16px;">{result}</span>
        #     </div>
        # """, unsafe_allow_html=True)
    st.write(result)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 Developed by Alpha Beta Gamma | Powered by TensorFlow & Hugging Face")
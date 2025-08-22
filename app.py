import streamlit as st
from groq import Groq
import easyocr
from PIL import Image

# Inicializar OCR una sola vez (lector multilenguaje)
reader = easyocr.Reader(['en', 'es'], gpu=False)

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        result = reader.readtext(image)
        # Unir todos los textos detectados
        text = " ".join([res[1] for res in result])
        return text if text.strip() else "⚠️ No se detectó texto en la imagen."
    except Exception as e:
        return f"⚠️ Error al procesar la imagen con EasyOCR: {e}"

def analyze_text_with_llm(text, language, api_key, model_name):
    client = Groq(api_key=api_key)

    prompt = f"Explica el siguiente texto detectado en una imagen en {language}:\n\n{text}"

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Eres un asistente experto en interpretación de textos extraídos de imágenes."},
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message["content"]

# --- UI Streamlit ---
st.title("🔎 OCR + LLM Analyzer")
st.write("Sube una imagen, se extraerá el texto automáticamente y luego se analizará con un LLM de Groq.")

# Input de API key
api_key_input = st.text_input("🔑 Ingresa tu API Key de Groq:", type="password")

# Modelo recomendado
model_name = "llama-3.3-70b-versatile"

# Idioma de salida
output_language = st.selectbox("🌐 Idioma de salida", ["español", "inglés"])

# Subir imagen
uploaded_file = st.file_uploader("📂 Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file and api_key_input:
    with st.spinner("📝 Extrayendo texto de la imagen..."):
        text = extract_text_from_image(uploaded_file)

    st.subheader("📝 Texto detectado")
    st.write(text)

    if text and not text.startswith("⚠️"):
        with st.spinner("🤖 Analizando con LLM de Groq..."):
            analysis = analyze_text_with_llm(text, language=output_language, api_key=api_key_input, model_name=model_name)

        st.subheader("📊 Análisis del texto")
        st.write(analysis)

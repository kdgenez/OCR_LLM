import streamlit as st
from groq import Groq
import easyocr
import numpy as np
from PIL import Image

# Inicializar OCR una sola vez (lector multilenguaje)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en", "es"], gpu=False)

reader = load_ocr_reader()

# Función para procesar la imagen con EasyOCR
def extract_text_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")  # Asegura formato
        image_np = np.array(image)  # Convertir a numpy array para EasyOCR
        result = reader.readtext(image_np)
        text = " ".join([res[1] for res in result])
        return text
    except Exception as e:
        st.error(f"⚠️ Error al procesar la imagen con EasyOCR: {e}")
        return ""

# Función para analizar texto con el LLM de Groq
def analyze_text_with_llm(text, language="es", api_key=None, model_name="llama-3.3-70b-versatile"):
    if not api_key:
        st.error("❌ Debes ingresar tu API Key de Groq.")
        return ""

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"Eres un asistente que analiza texto y responde en {language}."},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"⚠️ Error al llamar al modelo LLM: {e}")
        return ""

# Interfaz Streamlit
def main():
    st.title("📝 OCR + Análisis con LLM (Groq)")
    st.write("Sube una imagen con texto y el modelo de Groq la analizará.")

    # Input para API Key
    api_key_input = st.text_input("🔑 Ingresa tu API Key de Groq:", type="password")

    # Selección de idioma de salida
    output_language = st.selectbox("🌍 Selecciona idioma de salida:", ["es", "en"])

    # Subir imagen
    uploaded_file = st.file_uploader("📤 Sube una imagen", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Imagen subida", use_column_width=True)

        if st.button("🔎 Procesar imagen"):
            with st.spinner("Extrayendo texto de la imagen..."):
                text = extract_text_from_image(uploaded_file)

            if text:
                st.subheader("📝 Texto detectado")
                st.write(text)

                with st.spinner("Analizando texto con LLM..."):
                    analysis = analyze_text_with_llm(
                        text,
                        language=output_language,
                        api_key=api_key_input,
                        model_name="llama-3.3-70b-versatile"
                    )

                if analysis:
                    st.subheader("🤖 Análisis del texto")
                    st.write(analysis)

if __name__ == "__main__":
    main()

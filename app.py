import os
import io
import streamlit as st
from PIL import Image

# OCR
import easyocr

# LLM (Groq)
from groq import Groq

# ---------- Helpers ----------

@st.cache_resource(show_spinner=False)
def load_ocr_reader(langs=("es", "en")):
    return easyocr.Reader(list(langs), gpu=False, verbose=False, download_enabled=True)

@st.cache_data(show_spinner=False)
def run_ocr_on_image(img_bytes: bytes, langs):
    import numpy as np
    reader = load_ocr_reader(langs)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result = reader.readtext(np.asarray(image), detail=0, paragraph=True)
    return "\n".join(result)

def get_groq_client(api_key: str):
    if not api_key:
        return None
    return Groq(api_key=api_key)

def analyze_text_with_llm(text: str, language: str, api_key: str, model_name: str) -> str:
    client = get_groq_client(api_key)
    if client is None:
        raise RuntimeError("No se proporcionó una API Key válida de Groq.")

    system_prompt = (
        "Eres un asistente experto en análisis de texto extraído de imágenes. "
        "Debes resumir, explicar y dar contexto sobre el contenido, en un lenguaje claro."
    )

    user_prompt = f"""
    Analiza el siguiente texto detectado por OCR y devuelve:
    - Un resumen en {language} (máx 150 palabras)
    - Palabras clave principales
    - Posibles errores de OCR o partes confusas

    Texto:
    ---
    {text}
    ---
    """

    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.3,
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return completion.choices[0].message.content.strip()

# ---------- UI ----------

st.set_page_config(page_title="Analizador de Texto en Imágenes", page_icon="🖼️", layout="wide")

st.title("🖼️ Analizador de Texto en Imágenes (OCR + LLM)")
st.caption("Prototipo: OCR con EasyOCR + análisis con Groq LLM.")

with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input("🔑 Ingresa tu API Key de Groq", type="password")

    lang_choice = st.multiselect(
        "Idiomas OCR",
        options=["es", "en", "pt", "fr", "de", "it"],
        default=["es", "en"],
        help="Selecciona idiomas que podrían estar en la imagen.",
    )
    output_language = st.selectbox(
        "Idioma de la explicación",
        ["es", "en", "pt"],
        index=0
    )
    model_name = st.text_input(
        "Modelo LLM de Groq a usar",
        value="llama-3.1-70b-versatile",
        help="Especifica un modelo disponible en Groq (ej: llama-3.1-70b-versatile)"
    )

uploaded = st.file_uploader(
    "Sube una imagen (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if st.button("🔎 Analizar"):
    if not api_key_input:
        st.error("Debes ingresar una API Key de Groq.")
        st.stop()

    if uploaded is None:
        st.error("Sube una imagen para analizar.")
        st.stop()

    img_bytes = uploaded.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    st.subheader("🖼️ Imagen")
    st.image(image, use_column_width=True)

    st.info("Ejecutando OCR… puede tardar en la primera ejecución.")
    try:
        text = run_ocr_on_image(img_bytes, tuple(lang_choice))
    except Exception as e:
        st.exception(e)
        st.stop()

    st.subheader("📜 Texto detectado (OCR)")
    st.code(text if text.strip() else "(sin texto detectado)")

    st.subheader("🤖 Análisis con LLM (Groq)")
    try:
        analysis = analyze_text_with_llm(text, language=output_language, api_key=api_key_input, model_name=model_name)
        st.write(analysis)
    except Exception as e:
        st.error("Error usando el LLM.")
        st.exception(e)

st.markdown("---")
st.caption("Hecho con ❤️ usando Streamlit, EasyOCR y Groq.")

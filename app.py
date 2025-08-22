import os
import io
import streamlit as st
from PIL import Image

# OCR
import easyocr

# LLM (Groq)
from groq import Groq

# Env (opcional, para local)
from dotenv import load_dotenv

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

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY", None)
    if not api_key:
        return None
    return Groq(api_key=api_key)

def analyze_text_with_llm(text: str, language: str = "es") -> str:
    client = get_groq_client()
    if client is None:
        raise RuntimeError("No se encontró GROQ_API_KEY. Agrega tu clave en st.secrets o .env")

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
        model="llama-3.1-70b-versatile",
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
    st.markdown("---")
    st.subheader("🔐 Clave Groq")
    st.write("• Usa `st.secrets['GROQ_API_KEY']` en producción\n• O un archivo `.env` con `GROQ_API_KEY=...` para local")

uploaded = st.file_uploader(
    "Sube una imagen (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if st.button("🔎 Analizar"):
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
        analysis = analyze_text_with_llm(text, language=output_language)
        st.write(analysis)
    except Exception as e:
        st.error("Error usando el LLM.")
        st.exception(e)

st.markdown("---")
st.caption("Hecho con ❤️ usando Streamlit, EasyOCR y Groq.")

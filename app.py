import streamlit as st
from PIL import Image
import pytesseract
from groq import Groq

# ------------------------------------------------------------
# Función OCR: extraer texto de la imagen
# ------------------------------------------------------------
def extract_text_from_image(image: Image.Image) -> str:
    text = pytesseract.image_to_string(image, lang="eng+spa")
    return text.strip()

# ------------------------------------------------------------
# Función para analizar texto con Groq LLM
# ------------------------------------------------------------
def analyze_text_with_llm(text: str, api_key: str, model_name: str, language: str = "es") -> str:
    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"Eres un asistente que explica textos de imágenes de forma clara en {language}.",
                },
                {
                    "role": "user",
                    "content": f"Analiza y explica el siguiente texto extraído de una imagen:\n\n{text}",
                },
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ Error al consultar el modelo: {e}"

# ------------------------------------------------------------
# Interfaz en Streamlit
# -------

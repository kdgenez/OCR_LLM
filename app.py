import streamlit as st
from groq import Groq
from PIL import Image
import pytesseract

# =========================
# Función: Procesar imagen con OCR
# =========================
def extract_text_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image, lang="eng+spa")
        return text.strip()
    except Exception as e:
        return f"⚠️ Error al procesar la imagen: {str(e)}"

# =========================
# Función: Consultar LLM
# =========================
def analyze_text_with_llm(text, api_key, model_name="llama-3.3-70b-versatile", language="es"):
    try:
        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"Eres un asistente que explica texto en lenguaje claro en {language}."},
                {"role": "user", "content": f"Explica este texto:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=500,
        )

        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error al consultar el modelo LLM: {str(e)}"

# =========================
# Interfaz Streamlit
# =========================
st.set_page_config(page_title="Analizador de Texto en Imágenes", layout="wide")

# Siempre mostrar título e instrucciones
st.title("📄🔍 Analizador de Texto en Imágenes con OCR + LLM")
st.write("Sube una imagen y obtén una explicación del texto detectado usando un modelo de **Groq**.")

# Entrada API Key
api_key_input = st.sidebar.text_input("🔑 Ingresa tu API Key de Groq", type="password")

# Selección de idioma de salida
output_language = st.sidebar.selectbox("🌐 Idioma de salida", ["es", "en"])

# Subida de archivo
uploaded_file = st.file_uploader("📤 Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

if not api_key_input:
    st.info("👉 Por favor ingresa tu API Key de Groq en la barra lateral para continuar.")
else:
    if uploaded_file:
        st.image(uploaded_file, caption="📸 Imagen cargada", use_column_width=True)

        with st.spinner("⏳ Extrayendo texto con OCR..."):
            extracted_text = extract_text_from_image(uploaded_file)

        if extracted_text:
            st.subheader("📝 Texto detectado")
            st.write(extracted_text)

            with st.spinner("🤖 Analizando con LLM..."):
                analysis = analyze_text_with_llm(extracted_text, api_key=api_key_input, language=output_language)

            st.subheader("📖 Explicación generada")
            st.write(analysis)
        else:
            st.warning("⚠️ No se pudo extraer texto de la imagen.")
    else:
        st.info("📥 Sube una imagen para comenzar.")

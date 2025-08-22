# 🖼️ Analizador de Texto en Imágenes (OCR + LLM con Groq)

Un prototipo en **Streamlit** que conecta **EasyOCR** (para leer texto desde imágenes) con un **LLM de Groq** (para resumir y explicar el contenido). Ya no está limitado a facturas: sirve para **cualquier imagen que contenga texto**.

> **Entrada:** imagen JPG/PNG  
> **Salida:** texto OCR + explicación/resumen generado por LLM

---

## 🚀 Demo rápida (local)

1. Clona este repo o copia los 3 archivos: `app.py`, `requirements.txt`, `README.md`.
2. Crea y activa un entorno virtual (opcional pero recomendado).
3. Instala dependencias:

```bash
pip install -r requirements.txt
```

4. Crea un archivo `.env` con tu clave de Groq:

```bash
echo "GROQ_API_KEY=tu_clave_aqui" > .env
```

5. Ejecuta la app:

```bash
streamlit run app.py
```

Abre el enlace que te muestra Streamlit. Sube una imagen con texto.

> **Nota:** La primera ejecución puede demorar porque **EasyOCR** descarga modelos.

---

## ☁️ Despliegue en Streamlit Cloud

1. Sube tu repo a GitHub.
2. En **Streamlit Cloud**, crea una nueva app apuntando a `app.py`.
3. En **Settings → Secrets**, agrega:
   ```toml
   GROQ_API_KEY = "tu_clave_aqui"
   ```

---

## 🔑 Sobre la clave Groq

- La app intenta leer `st.secrets["GROQ_API_KEY"]` (ideal en Cloud).  
- Si no existe, carga `.env` y usa `GROQ_API_KEY` (ideal en local).  
- Consigue tu clave en el panel de Groq.

---

## 📄 Formatos de entrada

- **Imágenes**: `.jpg`, `.jpeg`, `.png`  
- PDF no está habilitado por defecto. Si lo necesitas, puedes integrar `pdf2image`.

---

## 🧠 Cómo funciona

1. **OCR (EasyOCR)** extrae el texto en bruto de la imagen.  
2. **Groq LLM** (por defecto `llama-3.1-70b-versatile`) devuelve:
   - Un **resumen** del contenido.  
   - **Palabras clave principales**.  
   - **Advertencias sobre posibles errores de OCR**.

---

## 🧪 Notas técnicas

- `@st.cache_resource` y `@st.cache_data` minimizan tiempos de carga del OCR.  
- El modelo de EasyOCR se carga en **CPU** (`gpu=False`) para mayor compatibilidad.  

---

## ⚠️ Limitaciones

- OCR puede confundirse con fotos borrosas, inclinadas o con bajo contraste.  
- El resumen depende del texto extraído; si el OCR falla, el análisis será limitado.  
- Este es un prototipo educativo.

---

## 🧩 Extensiones sugeridas

- Soporte de **PDF** con `pdf2image`.  
- Detección automática de idioma.  
- Exportar resultados a CSV o base de datos.  

---

## 📜 Licencia

MIT (ajústala según tu necesidad).

"""
Demo UI para el modelo de deteccion de foliculos (YOLOv8-seg)
Ejecutar localmente en tu laptop (con GPU RTX 5060) con:

    streamlit run demo_foliculos.py

Requisitos (si no los tenes ya en tu venv):
    pip install streamlit ultralytics opencv-python-headless pillow
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import cv2

# ---------- CONFIG ----------
MODEL_PATH = r"C:\Users\IA Concebir\foliculos-ia\runs\segment\train-3\weights\best.pt"

st.set_page_config(page_title="Foliculos AI - Demo", layout="wide")

st.title("🔬 Foliculos AI — Demo de deteccion")
st.caption("Segmentacion de foliculos en ultrasonido 2D (GE Voluson) con YOLOv8-seg")

# ---------- LOAD MODEL (cacheado para no recargarlo en cada interaccion) ----------
@st.cache_resource
def load_model(path):
    return YOLO(path)

if not os.path.exists(MODEL_PATH):
    st.error(f"No encuentro el modelo en:\n{MODEL_PATH}\n\nAjusta la variable MODEL_PATH en el script.")
    st.stop()

model = load_model(MODEL_PATH)

# ---------- SIDEBAR ----------
st.sidebar.header("Parametros")
conf = st.sidebar.slider("Confianza minima", 0.05, 0.95, 0.5, 0.05)
modo = st.sidebar.radio("Modo", ["Imagen", "Video"])

# ---------- IMAGEN ----------
if modo == "Imagen":
    archivo = st.file_uploader("Sube una imagen de ultrasonido", type=["png", "jpg", "jpeg", "bmp"])

    if archivo is not None:
        img = Image.open(archivo).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(img, width=380)

        with st.spinner("Corriendo inferencia..."):
            results = model.predict(np.array(img), conf=conf, verbose=False)
            r = results[0]
            plotted = r.plot(font_size=8, line_width=1)  # numpy array BGR con mascaras dibujadas
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Deteccion")
            st.image(plotted_rgb, width=380)

        n_foliculos = len(r.boxes) if r.boxes is not None else 0
        st.success(f"Foliculos detectados: {n_foliculos}")

        if n_foliculos > 0:
            st.subheader("Detalle por foliculo")
            data = []
            for i, box in enumerate(r.boxes):
                conf_val = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                ancho = xyxy[2] - xyxy[0]
                alto = xyxy[3] - xyxy[1]
                data.append({
                    "Foliculo": i + 1,
                    "Confianza": round(conf_val, 3),
                    "Ancho (px)": round(ancho, 1),
                    "Alto (px)": round(alto, 1),
                })
            st.table(data)

# ---------- VIDEO ----------
else:
    archivo = st.file_uploader("Sube un video de ultrasonido", type=["mp4", "avi", "mov"])
    usar_tracking = st.sidebar.checkbox("Activar tracking (ByteTrack)", value=True)

    if archivo is not None:
        # Guardar el video temporalmente porque YOLO necesita una ruta de archivo
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(archivo.read())
        input_path = tfile.name
        tfile.close()  # importante en Windows: liberar el handle antes de que YOLO/cv2 lo abra

        st.video(input_path)

        if st.button("Procesar video"):
            with st.spinner("Procesando video... esto puede tardar segun duracion/resolucion"):
                if usar_tracking:
                    results = model.track(
                        source=input_path,
                        conf=conf,
                        tracker="bytetrack.yaml",
                        save=True,
                        project=tempfile.gettempdir(),
                        name="foliculos_out",
                        exist_ok=True,
                    )
                else:
                    results = model.predict(
                        source=input_path,
                        conf=conf,
                        save=True,
                        project=tempfile.gettempdir(),
                        name="foliculos_out",
                        exist_ok=True,
                    )

            out_dir = os.path.join(tempfile.gettempdir(), "foliculos_out")
            # Buscar el archivo de video de salida generado por ultralytics
            salida = None
            if os.path.exists(out_dir):
                for f in os.listdir(out_dir):
                    if f.lower().endswith((".mp4", ".avi")):
                        salida = os.path.join(out_dir, f)
                        break

            if salida and os.path.exists(salida):
                st.success("Listo. Video procesado:")
                st.video(salida)
            else:
                st.warning("El video se proceso pero no encontre el archivo de salida esperado. Revisa la carpeta: " + out_dir)

        os.unlink(input_path) if False else None  # no borramos hasta que Streamlit libere el archivo

st.sidebar.markdown("---")
st.sidebar.caption("v1 — ajustar MODEL_PATH y umbrales segun evolucione el entrenamiento")
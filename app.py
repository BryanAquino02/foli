"""
Demo UI para el modelo de deteccion de foliculos (YOLOv8-seg)
Version adaptada para correr en Hugging Face Spaces (SDK: streamlit)

El modelo (best.pt) debe estar subido en la raiz del Space, junto a este archivo.
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import cv2

from calibration import mask_to_diameter_mm, fit_ellipse_diameter_mm, fit_ellipse_px, get_scale
from visualizacion import draw_measurements

# ---------- CONFIG ----------
# Ruta relativa: en HF Spaces el modelo debe subirse junto a app.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

# calibration.py lee las credenciales de Supabase de os.environ (para poder
# usarse también fuera de Streamlit, ej. desde auto_calibrate.py por
# consola). Acá las copiamos desde st.secrets para que ambos coincidan sin
# duplicar configuración en dos lugares.
for _key in ("SUPABASE_URL", "SUPABASE_ANON_KEY"):
    if _key in st.secrets and _key not in os.environ:
        os.environ[_key] = str(st.secrets[_key])

# Equipo en el que se calibró esta instancia (debe coincidir EXACTO con el
# perfil ya guardado en Supabase por auto_calibrate.py o la app de
# calibración manual). Se lee de los secrets de este deploy -- así el mismo
# app.py sirve para cualquier sede, solo cambia la configuración por deploy,
# no el código.
#
# En Streamlit Cloud / HF Spaces, agrega esto en "Secrets":
#   EQUIPO_MARCA = "GE"
#   EQUIPO_MODELO = "Voluson"
#   EQUIPO_TRANSDUCTOR = "6V1"
#   EQUIPO_PROFUNDIDAD_CM = 6.0
#
# Los valores de abajo son solo el respaldo para correrlo en tu compu local
# sin configurar secrets.
EQUIPO_MARCA = st.secrets.get("EQUIPO_MARCA", "GE")
EQUIPO_MODELO = st.secrets.get("EQUIPO_MODELO", "Voluson")
EQUIPO_TRANSDUCTOR = st.secrets.get("EQUIPO_TRANSDUCTOR", "6V1")
EQUIPO_PROFUNDIDAD_CM = float(st.secrets.get("EQUIPO_PROFUNDIDAD_CM", 6.0))

st.set_page_config(page_title="Foliculos AI - Demo", layout="wide")

st.title("🔬 Foliculos AI — Demo de deteccion")
st.caption("Segmentacion y medicion de foliculos en ultrasonido 2D (GE Voluson) con YOLOv8-seg")

# ---------- LOAD MODEL (cacheado para no recargarlo en cada interaccion) ----------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# ---------- LOAD ESCALA (cacheada, se busca UNA vez, no por imagen) ----------
@st.cache_resource
def load_escala():
    return get_scale(
        marca=EQUIPO_MARCA,
        modelo=EQUIPO_MODELO,
        profundidad_cm=EQUIPO_PROFUNDIDAD_CM,
        transductor=EQUIPO_TRANSDUCTOR,
    )

if not os.path.exists(MODEL_PATH):
    st.error(
        f"No encuentro el modelo en:\n{MODEL_PATH}\n\n"
        "Sube el archivo 'best.pt' a la raiz de este Space (junto a app.py)."
    )
    st.stop()

model = load_model(MODEL_PATH)

ESCALA_MM_PX = load_escala()
if ESCALA_MM_PX is None:
    st.error(
        f"No hay un perfil de calibración guardado para "
        f"{EQUIPO_MARCA} {EQUIPO_MODELO} ({EQUIPO_TRANSDUCTOR}, "
        f"D={EQUIPO_PROFUNDIDAD_CM}cm).\n\n"
        f"Este equipo necesita calibrarse primero con auto_calibrate.py "
        f"o la app de calibración manual antes de poder medir en mm."
    )
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("Parametros")
conf = st.sidebar.slider("Confianza minima", 0.05, 0.95, 0.5, 0.05)
modo = st.sidebar.radio("Modo", ["Imagen", "Video"])
st.sidebar.caption(f"Escala activa: {ESCALA_MM_PX:.4f} mm/px")

# ---------- IMAGEN ----------
if modo == "Imagen":
    archivo = st.file_uploader("Sube una imagen de ultrasonido", type=["png", "jpg", "jpeg", "bmp"])

    if archivo is not None:
        img = Image.open(archivo).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(img, use_container_width=True)

        with st.spinner("Corriendo inferencia..."):
            results = model.predict(np.array(img), conf=conf, verbose=False)
            r = results[0]

            # Por cada mascara segmentada: la reescalamos a las dimensiones
            # ORIGINALES de la imagen (masks.data viene en la resolucion interna
            # del modelo, ej. 448x640, no en el tamaño real de la imagen -- si no
            # se reescala, la elipse queda medida y dibujada en el lugar equivocado)
            orig_h, orig_w = r.orig_shape
            foliculos = []
            if r.masks is not None:
                for i, (mask, box) in enumerate(zip(r.masks.data, r.boxes)):
                    mask_np = mask.cpu().numpy().astype("uint8")
                    mask_np = cv2.resize(
                        mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                    ejes = fit_ellipse_diameter_mm(mask_np, ESCALA_MM_PX)
                    ellipse_px = fit_ellipse_px(mask_np)
                    diam_equiv_mm, _ = mask_to_diameter_mm(mask_np, ESCALA_MM_PX)

                    foliculos.append({
                        "id": i + 1,
                        "confianza": float(box.conf[0]),
                        "eje_mayor_mm": round(ejes["eje_mayor_mm"], 2) if ejes else None,
                        "eje_menor_mm": round(ejes["eje_menor_mm"], 2) if ejes else None,
                        "promedio_ejes_mm": round(ejes["promedio_mm"], 2) if ejes else None,
                        "diametro_equivalente_mm": round(diam_equiv_mm, 2),
                        "ellipse": ellipse_px,
                        "mask": mask_np,
                    })

            # Dibujamos las lineas de medicion (estilo caliper) sobre la imagen,
            # en vez del plot por defecto de ultralytics (que solo marca cajas)
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            plotted = draw_measurements(img_bgr, foliculos) if foliculos else img_bgr
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Deteccion y medicion")
            st.image(plotted_rgb, use_container_width=True)

        n_foliculos = len(foliculos)
        st.success(f"Foliculos detectados: {n_foliculos}")

        if n_foliculos > 0:
            st.subheader("Detalle por foliculo")
            data = []
            for f in foliculos:
                data.append({
                    "Foliculo": f["id"],
                    "Confianza": round(f["confianza"], 3),
                    "Eje mayor (mm)": f["eje_mayor_mm"],
                    "Eje menor (mm)": f["eje_menor_mm"],
                    "Promedio (mm)": f["promedio_ejes_mm"],
                    "Diametro equiv. (mm)": f["diametro_equivalente_mm"],
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
        tfile.close()  # liberar el handle antes de que YOLO/cv2 lo abra

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

        try:
            os.unlink(input_path)
        except OSError:
            pass

st.sidebar.markdown("---")
st.sidebar.caption("v1 — ajustar umbrales segun evolucione el entrenamiento")

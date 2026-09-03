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

from calibration import mask_to_diameter_mm, feret_diameters_mm, get_scale
from visualizacion import draw_measurements

# ---------- CONFIG ----------
# Ruta relativa: en HF Spaces el modelo debe subirse junto a app.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

# Ancho fijo (en px) para la previsualizacion de video (subido y procesado).
# Antes ocupaban el ancho completo de la columna, lo que se veia muy grande
# en la interfaz. Bajar este numero para reducirlos aun mas.
VIDEO_WIDTH = 480

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

# ---------- FUNCION COMPARTIDA: mismo pipeline que el modo Imagen ----------
# Toma un resultado (r) de ultralytics para UN frame y arma la lista de
# foliculos con medidas en mm, igual que en modo Imagen. Si track_ids viene
# (de model.track con persist=True), usamos ese id -- asi el mismo foliculo
# conserva su numero entre frames en vez de renumerarse cada vez.
def procesar_resultado(r, escala_mm_px):
    orig_h, orig_w = r.orig_shape
    foliculos = []
    if r.masks is None:
        return foliculos

    track_ids = None
    if r.boxes is not None and r.boxes.id is not None:
        track_ids = r.boxes.id.int().cpu().tolist()

    for i, (mask, box) in enumerate(zip(r.masks.data, r.boxes)):
        # Mismo bug de siempre: masks.data viene en resolucion del modelo,
        # hay que reescalar a la resolucion original del frame antes de medir.
        # OJO: el resize se hace en float, con interpolacion lineal, y
        # la mascara se binariza RECIEN despues del resize -- si se
        # binariza antes (uint8 + INTER_NEAREST), cada pixel del modelo
        # se estira en bloques cuadrados grandes y el contorno sale
        # poligonal/facetado en vez de curvo.
        mask_np = mask.cpu().numpy().astype("float32")
        mask_np = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_np = (mask_np > 0.5).astype("uint8")

        ejes = feret_diameters_mm(mask_np, escala_mm_px)
        diam_equiv_mm, _ = mask_to_diameter_mm(mask_np, escala_mm_px)

        foliculo_id = track_ids[i] if track_ids is not None else i + 1

        foliculos.append({
            "id": foliculo_id,
            "confianza": float(box.conf[0]),
            "eje_mayor_mm": round(ejes["eje_mayor_mm"], 2) if ejes else None,
            "eje_menor_mm": round(ejes["eje_menor_mm"], 2) if ejes else None,
            "promedio_ejes_mm": round(ejes["promedio_mm"], 2) if ejes else None,
            "diametro_equivalente_mm": round(diam_equiv_mm, 2),
            "p_mayor_1": ejes["p_mayor_1"] if ejes else None,
            "p_mayor_2": ejes["p_mayor_2"] if ejes else None,
            "p_menor_1": ejes["p_menor_1"] if ejes else None,
            "p_menor_2": ejes["p_menor_2"] if ejes else None,
            "mask": mask_np,
        })
    return foliculos
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
            foliculos = procesar_resultado(r, ESCALA_MM_PX)

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
    frame_skip = st.sidebar.slider(
        "Procesar 1 de cada N frames",
        min_value=1, max_value=15, value=5, step=1,
        help="N=1 procesa todos los frames (mas lento, mejor continuidad de tracking). "
             "N mas alto es mas rapido pero el tracker puede perder el ID si la sonda se mueve mucho entre frames procesados."
    )

    if archivo is not None:
        # Guardar el video temporalmente porque YOLO necesita una ruta de archivo
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(archivo.read())
        input_path = tfile.name
        tfile.close()  # liberar el handle antes de que YOLO/cv2 lo abra

        col_original, col_procesado = st.columns(2)

        with col_original:
            st.subheader("Original")
            st.video(input_path, width=VIDEO_WIDTH)

        # Placeholder del lado derecho: antes de procesar solo muestra un
        # aviso; se reemplaza con el video ya procesado mas abajo.
        with col_procesado:
            st.subheader("Deteccion y medicion")
            resultado_placeholder = st.empty()
            resultado_placeholder.info("Presiona 'Procesar video' para ver el resultado aqui.")

        if st.button("Procesar video"):
            cap = cv2.VideoCapture(input_path)
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Nombre unico por corrida (evita que una corrida anterior que no
            # cerro bien el writer, o que sigue corriendo, choque con esta).
            out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
            os.close(out_fd)
            writer = None  # se crea recien con el primer frame (ya sabemos su tamaño)
            # Usamos imageio + imageio-ffmpeg (trae su propio binario de ffmpeg
            # empaquetado en el .whl, no depende de nada del sistema operativo)
            # en vez de cv2.VideoWriter, porque el codec 'mp4v' de OpenCV genera
            # archivos que muchos navegadores (Chrome/Safari) no pueden reproducir
            # via <video> -- se necesita H.264 (libx264) en un mp4 bien formado.
            import imageio

            # fps efectivo del video de salida: si saltamos frames, el video de
            # salida tambien corre mas lento en frames-procesados-por-segundo,
            # asi que ajustamos el fps de escritura para que la duracion visual
            # se mantenga parecida al video original.
            fps_out = max(fps_in / frame_skip, 1)

            frame_idx = 0
            procesados = 0

            # Acumula, por ID de foliculo, la medida MAXIMA vista en cualquier
            # frame (no el ultimo frame). Solo tiene sentido si el tracking
            # esta activo, porque ahi el ID se mantiene entre frames -- sin
            # tracking cada frame renumera los foliculos desde 1 y agruparlos
            # por "id" mezclaria foliculos distintos.
            foliculos_max = {}

            # Todo lo de esta corrida (progreso, spinner, video final) va
            # dentro de la columna derecha, para que quede al lado del
            # video original en vez de debajo de todo.
            resultado_placeholder.empty()
            progress = col_procesado.progress(0.0)
            status = col_procesado.empty()

            # stream=True: en vez de que ultralytics guarde un video completo con
            # su propio r.plot() (sin mm), iteramos resultado por resultado y
            # aplicamos el mismo pipeline de medicion que en modo Imagen antes de
            # escribir el frame de salida.
            if usar_tracking:
                fuente_resultados = model.track(
                    source=input_path,
                    conf=conf,
                    tracker="bytetrack.yaml",
                    persist=True,
                    stream=True,
                    verbose=False,
                    vid_stride=frame_skip,
                )
            else:
                fuente_resultados = model.predict(
                    source=input_path,
                    conf=conf,
                    stream=True,
                    verbose=False,
                    vid_stride=frame_skip,
                )

            try:
                with col_procesado, st.spinner("Procesando video... esto puede tardar segun duracion/resolucion"):
                    for r in fuente_resultados:
                        frame_bgr = r.orig_img  # ya viene en BGR (formato de cv2/ultralytics)
                        foliculos = procesar_resultado(r, ESCALA_MM_PX)
                        plotted = draw_measurements(frame_bgr, foliculos) if foliculos else frame_bgr

                        if usar_tracking:
                            for f in foliculos:
                                fid = f["id"]
                                entry = foliculos_max.setdefault(fid, {
                                    "eje_mayor_mm": None,
                                    "eje_menor_mm": None,
                                    "promedio_ejes_mm": None,
                                    "diametro_equivalente_mm": None,
                                    "confianza_max": None,
                                })
                                for campo in (
                                    "eje_mayor_mm", "eje_menor_mm",
                                    "promedio_ejes_mm", "diametro_equivalente_mm",
                                ):
                                    valor = f[campo]
                                    if valor is not None and (entry[campo] is None or valor > entry[campo]):
                                        entry[campo] = valor
                                if entry["confianza_max"] is None or f["confianza"] > entry["confianza_max"]:
                                    entry["confianza_max"] = f["confianza"]

                        # imageio espera frames en RGB, no BGR (formato nativo de cv2/ultralytics)
                        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

                        if writer is None:
                            # No pasamos "-pix_fmt yuv420p" a mano: imageio-ffmpeg ya
                            # lo agrega por defecto con codec libx264, y pasarlo de
                            # nuevo generaba el warning "Multiple -pix_fmt options
                            # specified" en los logs (no rompia nada, pero ensuciaba).
                            writer = imageio.get_writer(
                                out_path,
                                fps=fps_out,
                                codec="libx264",
                                quality=8,
                                macro_block_size=1,  # evita que imageio recorte/reescale por multiplos de 16
                            )

                        writer.append_data(plotted_rgb)

                        procesados += 1
                        frame_idx += frame_skip
                        if total_frames:
                            progress.progress(min(frame_idx / total_frames, 1.0))
                        status.caption(f"Frames procesados: {procesados}")
            finally:
                # Pase lo que pase (exito, excepcion, video vacio), liberamos el
                # proceso de ffmpeg. Si esto no se cierra, la proxima corrida
                # puede arrastrar procesos colgados y logs raros de ffmpeg.
                if writer is not None:
                    writer.close()

            with col_procesado:
                # Limpiamos la barra de progreso y el contador de frames: una
                # vez listo, ya no aportan nada y solo ensucian la vista.
                progress.empty()
                status.empty()

                if procesados > 0:
                    st.video(out_path, width=VIDEO_WIDTH)
                else:
                    st.warning("No se detecto ningun frame para escribir. Revisa el video de entrada.")

            if procesados > 0:
                if usar_tracking and foliculos_max:
                    st.subheader("Detalle por foliculo (medida maxima detectada en el video)")
                    data = []
                    for fid in sorted(foliculos_max.keys()):
                        m = foliculos_max[fid]
                        data.append({
                            "Foliculo": fid,
                            "Confianza max.": round(m["confianza_max"], 3) if m["confianza_max"] is not None else None,
                            "Eje mayor (mm)": m["eje_mayor_mm"],
                            "Eje menor (mm)": m["eje_menor_mm"],
                            "Promedio (mm)": m["promedio_ejes_mm"],
                            "Diametro equiv. (mm)": m["diametro_equivalente_mm"],
                        })
                    st.table(data)
                elif not usar_tracking:
                    st.info(
                        "Activa 'Activar tracking (ByteTrack)' en la barra lateral para "
                        "ver la tabla de medida maxima por foliculo. Sin tracking, el "
                        "ID de cada foliculo se reinicia en cada frame y no se puede "
                        "seguir su medida a lo largo del video."
                    )

        try:
            os.unlink(input_path)
        except OSError:
            pass

st.sidebar.markdown("---")
st.sidebar.caption("v1 — ajustar umbrales segun evolucione el entrenamiento")

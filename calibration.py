"""
calibration.py

Conecta el pipeline de detección de folículos (YOLOv8-seg) con los perfiles
de calibración guardados por la app de calibración (Supabase).

Instalación:
    pip install supabase opencv-python-headless numpy

Uso típico dentro de tu app de Streamlit / script de inferencia:

    from calibration import list_profiles, get_scale, mask_to_diameter_mm

    perfiles = list_profiles()
    # ... usuario elige uno en un selectbox de Streamlit ...
    escala = perfil_elegido["escala_mm_px"]

    for mask in resultados_yolo.masks.data:  # una máscara por folículo
        mask_np = mask.cpu().numpy().astype("uint8")
        diam_mm, diam_px = mask_to_diameter_mm(mask_np, escala)
        print(f"Folículo: {diam_mm:.2f} mm ({diam_px:.1f} px)")
"""

import os
import math
from typing import Optional

import cv2
import numpy as np
from supabase import create_client, Client

TABLE = "calibration_profiles"


def get_client() -> Client:
    """
    Crea el cliente de Supabase.
    Toma las credenciales de variables de entorno (recomendado) o,
    si no existen, de las constantes de abajo (solo para pruebas rápidas).
    """
    url = os.environ.get("SUPABASE_URL", "https://TU-PROYECTO.supabase.co")
    key = os.environ.get("SUPABASE_ANON_KEY", "TU-ANON-KEY-PUBLICA")
    if "TU-PROYECTO" in url:
        raise RuntimeError(
            "Configura SUPABASE_URL y SUPABASE_ANON_KEY como variables de "
            "entorno (o edita las constantes en calibration.py)."
        )
    return create_client(url, key)


def list_profiles() -> list[dict]:
    """Devuelve todos los perfiles de calibración guardados, más recientes primero."""
    supa = get_client()
    res = (
        supa.table(TABLE)
        .select("*")
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


def get_scale(
    marca: str,
    modelo: str,
    profundidad_cm: float,
    transductor: str = "",
) -> Optional[float]:
    """
    Busca el perfil exacto (marca+modelo+transductor+profundidad) y devuelve
    escala_mm_px, o None si no existe ese perfil todavía.
    """
    supa = get_client()
    res = (
        supa.table(TABLE)
        .select("escala_mm_px")
        .eq("marca", marca)
        .eq("modelo", modelo)
        .eq("transductor", transductor or "")
        .eq("profundidad_cm", profundidad_cm)
        .limit(1)
        .execute()
    )
    if res.data:
        return float(res.data[0]["escala_mm_px"])
    return None


def mask_to_diameter_mm(mask: np.ndarray, escala_mm_px: float) -> tuple[float, float]:
    """
    Convierte una máscara binaria de segmentación (0/1 o 0/255) al diámetro
    real del folículo en mm.

    Usa el diámetro equivalente por área (2*sqrt(area/pi)), que es la métrica
    estándar en literatura de foliculometría cuando el folículo no es un
    círculo perfecto. Si prefieres el eje mayor de una elipse ajustada
    (más parecido a cómo mide manualmente un ecografista con 2 calipers),
    usa fit_ellipse_diameter_mm en su lugar.

    Devuelve: (diametro_mm, diametro_px)
    """
    mask_bin = (mask > 0).astype(np.uint8)
    area_px = float(mask_bin.sum())
    if area_px == 0:
        return 0.0, 0.0
    diam_px = 2 * math.sqrt(area_px / math.pi)
    diam_mm = diam_px * escala_mm_px
    return diam_mm, diam_px


def fit_ellipse_px(mask: np.ndarray) -> Optional[tuple]:
    """
    Ajusta una elipse al contorno del folículo y devuelve la geometría cruda
    en píxeles: ((cx, cy), (eje_menor_px, eje_mayor_px), angulo_grados).
    Es el mismo formato que devuelve cv2.fitEllipse -- útil tanto para
    convertir a mm como para dibujar la elipse/eje sobre la imagen.

    Devuelve None si el contorno es muy pequeño para ajustar una elipse.
    """
    mask_bin = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None  # fitEllipse necesita al menos 5 puntos
    return cv2.fitEllipse(largest)


def fit_ellipse_diameter_mm(
    mask: np.ndarray, escala_mm_px: float
) -> Optional[dict]:
    """
    Alternativa más clínica: ajusta una elipse al contorno del folículo y
    devuelve eje mayor y menor en mm (como el ecografista mide 2 diámetros
    perpendiculares con el caliper).

    Devuelve dict con 'eje_mayor_mm', 'eje_menor_mm', 'promedio_mm' o None
    si el contorno es muy pequeño para ajustar elipse.
    """
    ellipse = fit_ellipse_px(mask)
    if ellipse is None:
        return None

    (_, _), (eje_menor_px, eje_mayor_px), _ = ellipse
    eje_mayor_mm = eje_mayor_px * escala_mm_px
    eje_menor_mm = eje_menor_px * escala_mm_px
    return {
        "eje_mayor_mm": eje_mayor_mm,
        "eje_menor_mm": eje_menor_mm,
        "promedio_mm": (eje_mayor_mm + eje_menor_mm) / 2,
    }

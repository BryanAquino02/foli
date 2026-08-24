"""
visualizacion.py

Dibuja sobre la imagen, para cada folículo detectado, el eje de medición
(como la línea de caliper del ecógrafo) y su diámetro en mm -- para que la
IA no solo detecte el folículo, sino que "muestre cómo lo midió".

Uso:
    from visualizacion import draw_measurements

    imagen_anotada = draw_measurements(imagen_original_bgr, foliculos_geometria)
    cv2.imwrite("resultado.jpg", imagen_anotada)
"""

import math

import cv2
import numpy as np

# Colores por categoría de tamaño (BGR, formato OpenCV)
COLOR_PEQUENO = (219, 152, 52)   # azul
COLOR_MEDIANO = (26, 184, 240)   # ámbar
COLOR_GRANDE = (60, 60, 230)     # rojo -- folículo dominante / maduro

COLOR_TEXT_BG = (26, 15, 10)
COLOR_TEXT = (245, 245, 245)

# Umbrales en mm (usando el promedio de los 2 ejes, como mide un ecografista).
# Ajusta estos 2 números al protocolo clínico real de Concebir.
UMBRAL_PEQUENO_MEDIANO = 10.0
UMBRAL_MEDIANO_GRANDE = 16.0


def _color_por_tamano(diametro_mm: float) -> tuple:
    if diametro_mm < UMBRAL_PEQUENO_MEDIANO:
        return COLOR_PEQUENO
    elif diametro_mm < UMBRAL_MEDIANO_GRANDE:
        return COLOR_MEDIANO
    else:
        return COLOR_GRANDE


def _ellipse_axis_endpoints(center, axis_len, angle_deg):
    """Calcula los 2 puntos extremos de un eje de la elipse (mayor o menor)."""
    angle_rad = math.radians(angle_deg)
    dx = (axis_len / 2) * math.sin(angle_rad)
    dy = -(axis_len / 2) * math.cos(angle_rad)
    cx, cy = center
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    return p1, p2


def draw_measurements(image_bgr: np.ndarray, foliculos: list[dict]) -> np.ndarray:
    """
    image_bgr: imagen original (formato OpenCV, BGR) sobre la que dibujar.
    foliculos: lista de dicts, cada uno con:
        - 'ellipse': tupla cruda de cv2.fitEllipse ((cx,cy),(menor,mayor),angulo)
        - 'eje_mayor_mm', 'eje_menor_mm': ya convertidos a mm
        - 'promedio_ejes_mm' (opcional): si no viene, se calcula del mayor/menor

    Cada folículo se dibuja con un color según su tamaño (pequeño/mediano/
    grande) -- útil para identificar de un vistazo el folículo dominante.

    Devuelve una COPIA de la imagen con las líneas y textos dibujados
    (no modifica la imagen original).
    """
    out = image_bgr.copy()

    for f in foliculos:
        ellipse = f.get("ellipse")
        if ellipse is None:
            continue

        center, (eje_menor_px, eje_mayor_px), angle = ellipse
        center_int = (int(round(center[0])), int(round(center[1])))

        promedio_mm = f.get("promedio_ejes_mm")
        if promedio_mm is None:
            promedio_mm = (f["eje_mayor_mm"] + f["eje_menor_mm"]) / 2
        color = _color_por_tamano(promedio_mm)

        # contorno de la elipse ajustada, en el color de su categoría de tamaño
        cv2.ellipse(out, ellipse, color, 2, cv2.LINE_AA)

        # eje mayor, dibujado como la línea de caliper (punteada, cruces en
        # los extremos) -- mismo color que el contorno, para que la
        # categoría de tamaño se lea de un vistazo
        p1, p2 = _ellipse_axis_endpoints(center, eje_mayor_px, angle)
        _draw_dotted_line(out, p1, p2, color)
        _draw_cross(out, p1, color)
        _draw_cross(out, p2, color)

        # etiqueta con el diámetro mayor en mm
        label = f"{f['eje_mayor_mm']:.1f} mm"
        _draw_label(out, label, (center_int[0] + 8, center_int[1] - 8), color)

    return out


def _draw_cross(img, point, color, size=6, thickness=2):
    x, y = point
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def _draw_dotted_line(img, p1, p2, color, gap=6, thickness=1):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    n_dots = max(int(dist / gap), 1)
    for i in range(n_dots + 1):
        t = i / n_dots
        x = int(round(p1[0] + (p2[0] - p1[0]) * t))
        y = int(round(p1[1] + (p2[1] - p1[1]) * t))
        cv2.circle(img, (x, y), thickness, color, -1, cv2.LINE_AA)


def _draw_label(img, text, origin, accent_color, font_scale=0.45, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        img, (x - 3, y - th - 4), (x + tw + 3, y + 4), COLOR_TEXT_BG, -1
    )
    cv2.rectangle(
        img, (x - 3, y - th - 4), (x + tw + 3, y + 4), accent_color, 1
    )
    cv2.putText(
        img, text, (x, y), font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA
    )

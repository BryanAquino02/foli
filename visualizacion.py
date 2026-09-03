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

# Opacidad del relleno de la máscara (0 = invisible, 1 = sólido).
# 0.3-0.4 se ve bien sobre ultrasonido sin tapar la textura de fondo.
ALPHA_RELLENO = 0.35


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


def _rellenar_mascara_transparente(image_bgr, mask, color_bgr, alpha=ALPHA_RELLENO):
    """
    Rellena el área de una máscara con un color semitransparente, mezclando
    con la imagen de fondo solo en esa región (el resto de la imagen no se
    toca).

    image_bgr: imagen (BGR) sobre la que se pinta -- YA en la resolución
        original (ver bug #4 del README: hay que hacer cv2.resize del
        mask con r.orig_shape ANTES de llegar aquí, si no el relleno cae
        en el lugar equivocado).
    mask: máscara binaria del folículo (0/1 o 0/255), misma resolución
        que image_bgr.
    color_bgr: tupla (B, G, R).
    alpha: opacidad del relleno.
    """
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return image_bgr

    overlay = image_bgr.copy()
    overlay[mask_bool] = color_bgr

    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

    out = image_bgr.copy()
    out[mask_bool] = blended[mask_bool]
    return out


def draw_measurements(
    image_bgr: np.ndarray,
    foliculos: list[dict],
    mostrar_relleno: bool = True,
    mostrar_contorno: bool = True,
    mostrar_eje_mayor: bool = True,
    mostrar_eje_menor: bool = True,
    mostrar_etiqueta: bool = True,
) -> np.ndarray:
    """
    ... (mismo docstring que antes) ...

    mostrar_relleno / mostrar_contorno / mostrar_eje_mayor / mostrar_eje_menor /
    mostrar_etiqueta: activan o desactivan cada capa visual de forma
    independiente, para poder ver solo lo que se necesita en cada momento.
    """
    out = image_bgr.copy()

    for f in foliculos:
        p_mayor_1, p_mayor_2 = f.get("p_mayor_1"), f.get("p_mayor_2")
        if p_mayor_1 is None or p_mayor_2 is None:
            continue

        promedio_mm = f.get("promedio_ejes_mm")
        if promedio_mm is None:
            promedio_mm = (f["eje_mayor_mm"] + f["eje_menor_mm"]) / 2
        color = _color_por_tamano(promedio_mm)

        mask = f.get("mask")

        if mostrar_relleno and mask is not None:
            out = _rellenar_mascara_transparente(out, mask, color)

        if mostrar_contorno and mask is not None:
            mask_bin = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, (235, 235, 235), 1, cv2.LINE_AA)

        p1 = (int(round(p_mayor_1[0])), int(round(p_mayor_1[1])))
        p2 = (int(round(p_mayor_2[0])), int(round(p_mayor_2[1])))

        if mostrar_eje_mayor:
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
            _draw_cross(out, p1, color)
            _draw_cross(out, p2, color)

        if mostrar_eje_menor:
            p_menor_1, p_menor_2 = f.get("p_menor_1"), f.get("p_menor_2")
            if p_menor_1 is not None and p_menor_2 is not None:
                q1 = (int(round(p_menor_1[0])), int(round(p_menor_1[1])))
                q2 = (int(round(p_menor_2[0])), int(round(p_menor_2[1])))
                _draw_dotted_line(out, q1, q2, color)

        if mostrar_etiqueta:
            center = (int(round((p1[0]+p2[0])/2)), int(round((p1[1]+p2[1])/2)))
            label = f"{f['eje_mayor_mm']:.1f} mm"
            _draw_label(out, label, (center[0] + 8, center[1] - 8), color)

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

"""
Serializador para pasar los resultados de procesar_resultado() al visor
Three.js (foliculos_3d_viewer.html) via streamlit.components.v1.html.

No toca tu pipeline de deteccion: solo limpia el dict (saca 'mask', que es
un np.ndarray y no es JSON-serializable) y arma el string listo para
inyectar en el <script> del componente.
"""

import json


def color_por_tamano_hex(diametro_mm):
    """Misma logica de umbrales que usas en visualizacion_3d.py / visualizacion.py,
    pero en hex para que Three.js la use directo (MeshStandardMaterial quiere hex/int)."""
    if diametro_mm < 10:
        return "#3b82f6"   # azul
    elif diametro_mm < 16:
        return "#f59e0b"   # ambar
    else:
        return "#ef4444"   # rojo


def foliculos_a_json(foliculos):
    """
    foliculos: la lista que devuelve procesar_resultado(r, escala_mm_px).
    Devuelve un string JSON (no un dict) listo para inyectar en el HTML.
    """
    limpio = []
    for f in foliculos:
        if not f.get("contorno_mm"):
            continue  # sin contorno no hay nada que extruir

        limpio.append({
            "id": f["id"],
            "contorno_mm": f["contorno_mm"],          # lista de (x_mm, y_mm), ya en orden de contorno
            "centro_x_mm": f["centro_x_mm"],
            "centro_y_mm": f["centro_y_mm"],
            "diametro_mm": f["diametro_equivalente_mm"],
            "color": color_por_tamano_hex(f["diametro_equivalente_mm"]),
        })

    return json.dumps(limpio)


def frame_a_html(foliculos, html_template_path, height_mm_escena=None):
    """
    Helper de conveniencia: lee la plantilla HTML del visor, inyecta los
    datos del frame actual y devuelve el HTML final listo para
    components.html(...).

    html_template_path: ruta a foliculos_3d_viewer.html
    """
    with open(html_template_path, "r", encoding="utf-8") as fh:
        template = fh.read()

    datos_json = foliculos_a_json(foliculos)
    html_final = template.replace("__DATOS_FOLICULOS__", datos_json)
    return html_final
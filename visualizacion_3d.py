import plotly.graph_objects as go
import numpy as np


def color_por_tamano(diametro_mm):
    if diametro_mm < 10:
        return "blue"
    elif diametro_mm < 16:
        return "orange"
    else:
        return "red"


def generar_blob(contorno_mm, cx_mm, cy_mm, resolucion_z=8, resolucion_angular=40):
    """
    Extruye el contorno real en una forma tipo lente/burbuja.
    Remuestrea el contorno por angulo (coordenadas polares) para evitar
    cruces, y CIERRA el loop angular repitiendo el primer punto al final
    para que no quede una costura visible.
    """
    if len(contorno_mm) < 3:
        return None, None, None

    xs_raw = np.array([p[0] - cx_mm for p in contorno_mm])
    ys_raw = np.array([-(p[1] - cy_mm) for p in contorno_mm])

    angulos_raw = np.arctan2(ys_raw, xs_raw)
    radios_raw = np.sqrt(xs_raw**2 + ys_raw**2)

    orden = np.argsort(angulos_raw)
    angulos_ord = angulos_raw[orden]
    radios_ord = radios_raw[orden]

    # grilla angular CERRADA: ultimo punto = primero + 2*pi
    angulos_grid = np.linspace(-np.pi, np.pi, resolucion_angular, endpoint=True)
    radios_interp = np.interp(angulos_grid, angulos_ord, radios_ord, period=2 * np.pi)

    xs = radios_interp * np.cos(angulos_grid)
    ys = radios_interp * np.sin(angulos_grid)

    radio_max = radios_interp.max()
    if radio_max == 0:
        return None, None, None

    niveles_z = np.linspace(0, 1, resolucion_z)

    X, Y, Z = [], [], []
    for nivel in niveles_z:
        factor = 1 - nivel
        z_val = radio_max * np.sqrt(max(0, 1 - factor**2))
        X.append(cx_mm + xs * factor)
        Y.append(cy_mm + ys * factor)
        Z.append(np.full_like(xs, z_val))

    return np.array(X), np.array(Y), np.array(Z)

def mostrar_foliculos_3d(foliculos):
    fig = go.Figure()

    for f in foliculos:
        diametro = f["diametro_equivalente_mm"]
        contorno_mm = f.get("contorno_mm")
        if not contorno_mm:
            continue

        X, Y, Z = generar_blob(contorno_mm, f["centro_x_mm"], f["centro_y_mm"])
        if X is None:
            continue

        color = color_por_tamano(diametro)
        # mitad superior (abombada) + mitad inferior (espejo, Z negativo) para dar volumen completo
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, color], [1, color]],
            showscale=False, opacity=0.85,
            hovertemplate=f"Folículo: {diametro:.1f}mm<extra></extra>"
        ))
        fig.add_trace(go.Surface(
            x=X, y=Y, z=-Z,
            colorscale=[[0, color], [1, color]],
            showscale=False, opacity=0.85,
            hovertemplate=f"Folículo: {diametro:.1f}mm<extra></extra>"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="mm", yaxis_title="mm", zaxis_title="mm",
            aspectmode="data",
            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(x=0, y=0, z=2.5)  # vista desde arriba, como la ecografia
            )
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title="Folículos detectados (forma, posición y tamaño reales)"
    )
    return fig
import numpy as np
import plotly.graph_objects as go
import qrcode
from PIL import Image


def create_3d_voxel_plot(binary_grid, thickness=5):
    """Optimized 3D Voxel Plotter"""
    rows, cols = binary_grid.shape
    step = 2 if (rows > 50 or cols > 50) else 1

    x_verts = []
    y_verts = []
    z_verts = []
    i_indices = []
    j_indices = []
    k_indices = []
    vertex_count = 0

    for r in range(0, rows, step):
        for c in range(0, cols, step):
            if binary_grid[r, c] > 0.5:
                x, y = c, rows - 1 - r
                x_verts.extend([x, x+step, x+step, x,   x, x+step, x+step, x])
                y_verts.extend([y, y, y+step, y+step,   y, y, y+step, y+step])
                z_verts.extend([0, 0, 0, 0,       thickness,
                               thickness, thickness, thickness])

                cube_indices = [
                    0, 2, 1,   0, 3, 2,  4, 5, 6,   4, 6, 7,
                    0, 1, 5,   0, 5, 4,  2, 3, 7,   2, 7, 6,
                    0, 4, 7,   0, 7, 3,  1, 2, 6,   1, 6, 5
                ]
                i_indices.extend(
                    [idx + vertex_count for idx in cube_indices[0::3]])
                j_indices.extend(
                    [idx + vertex_count for idx in cube_indices[1::3]])
                k_indices.extend(
                    [idx + vertex_count for idx in cube_indices[2::3]])
                vertex_count += 8

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x_verts, y=y_verts, z=z_verts,
            i=i_indices, j=j_indices, k=k_indices,
            color='#3b82f6', opacity=1.0, flatshading=True,
            lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.1)
        )
    ])
    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(
        visible=False), bgcolor='rgba(0,0,0,0)'), margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def generate_ar_qrcode(url="https://neuro-topopt-aak.streamlit.app/"):
    """Generates QR code for live app URL"""
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

import qrcode
from PIL import Image
import numpy as np
import plotly.graph_objects as go


def create_3d_voxel_plot(binary_grid, thickness=5):
    """
    Converts a 2D binary grid into an interactive 3D Voxel Mesh for Plotly.
    """
    rows, cols = binary_grid.shape

    # Lists to store mesh data
    x_verts = []
    y_verts = []
    z_verts = []

    i_indices = []
    j_indices = []
    k_indices = []

    vertex_count = 0

    # Iterate through the grid to find material
    for r in range(rows):
        for c in range(cols):
            if binary_grid[r, c] > 0.5:  # If material exists
                # Coordinate mapping (Flip Y to match engineering view)
                x, y = c, rows - 1 - r

                # Define 8 vertices of a cube (voxel)
                # Base (z=0) and Top (z=thickness)
                # p0-p3 (Base), p4-p7 (Top)

                # Append 8 vertices
                x_verts.extend([x, x+1, x+1, x,   x, x+1, x+1, x])
                y_verts.extend([y, y, y+1, y+1,   y, y, y+1, y+1])
                z_verts.extend([0, 0, 0, 0,       thickness,
                               thickness, thickness, thickness])

                # Define 12 triangles (2 per face, 6 faces)
                # Standard Cube triangulation indices relative to current cube
                cube_indices = [
                    0, 2, 1,   0, 3, 2,  # Bottom
                    4, 5, 6,   4, 6, 7,  # Top
                    0, 1, 5,   0, 5, 4,  # Front
                    2, 3, 7,   2, 7, 6,  # Back
                    0, 4, 7,   0, 7, 3,  # Left
                    1, 2, 6,   1, 6, 5   # Right
                ]

                # Offset indices by current vertex count and add to main list
                i_indices.extend(
                    [idx + vertex_count for idx in cube_indices[0::3]])
                j_indices.extend(
                    [idx + vertex_count for idx in cube_indices[1::3]])
                k_indices.extend(
                    [idx + vertex_count for idx in cube_indices[2::3]])

                vertex_count += 8

    # Create the Mesh3d object
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x_verts,
            y=y_verts,
            z=z_verts,
            i=i_indices,
            j=j_indices,
            k=k_indices,
            color='#3b82f6',  # Professional Engineering Blue
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.5, diffuse=1.0)
        )
    ])

    # Engineering Layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',  # Keeps true proportions
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'  # Transparent background
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def generate_ar_qrcode(url="https://github.com/AkberAliKhan/NeuroTopOpt"):
    """
    Generates a QR code. In a real deployed app, this would link to the .GLB file.
    For local demo, it links to your Portfolio/GitHub.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

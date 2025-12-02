import io
import numpy as np
import ezdxf
from stl import mesh


def generate_dxf(grid_data, threshold=0.5):
    doc = ezdxf.new()
    msp = doc.modelspace()
    rows, cols = grid_data.shape

    for r in range(rows):
        for c in range(cols):
            if grid_data[r, c] > threshold:
                points = [(c, -r), (c + 1, -r), (c + 1, -r - 1), (c, -r - 1)]
                msp.add_lwpolyline(points, close=True)

    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode()


def generate_stl(grid_data, threshold=0.5, thickness=5):
    rows, cols = grid_data.shape
    faces = []
    for r in range(0, rows):
        for c in range(0, cols):
            if grid_data[r, c] > threshold:
                x, y = c, rows - 1 - r
                p0, p1, p2, p3 = [x, y, 0], [
                    x+1, y, 0], [x+1, y+1, 0], [x, y+1, 0]
                p4, p5, p6, p7 = [x, y, thickness], [
                    x+1, y, thickness], [x+1, y+1, thickness], [x, y+1, thickness]
                faces.extend([[p0, p2, p1], [p0, p3, p2], [p4, p5, p6], [p4, p6, p7],
                              [p0, p1, p5], [p0, p5, p4], [
                                  p2, p3, p7], [p2, p7, p6],
                              [p0, p4, p7], [p0, p7, p3], [p1, p2, p6], [p1, p6, p5]])
    np_faces = np.array(faces)
    surface = mesh.Mesh(np.zeros(np_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(np_faces):
        for j in range(3):
            surface.vectors[i][j] = f[j]
    buffer = io.BytesIO()
    surface.save('temp.stl', fh=buffer)
    buffer.seek(0)
    return buffer.getvalue()

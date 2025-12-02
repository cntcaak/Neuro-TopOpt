import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import ezdxf
from stl import mesh

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Neuro-TopOpt Ultimate",
                   layout="wide", page_icon="🏗️")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stMetric { background-color: #262730; padding: 15px; border-radius: 5px; border: 1px solid #4f4f4f; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Neuro-TopOpt: AI-to-Manufacturing Pipeline")
st.markdown("### Ultimate Edition | AI Design -> CAD -> 3D Print")

# --- HELPER FUNCTIONS FOR EXPORT ---


def generate_dxf(grid_data, threshold=0.5):
    """Converts the binary grid into a DXF file (2D Polygons)."""
    doc = ezdxf.new()
    msp = doc.modelspace()

    rows, cols = grid_data.shape

    # Iterate through grid
    for r in range(rows):
        for c in range(cols):
            # If material exists (AI confidence > threshold)
            if grid_data[r, c] > threshold:
                # Draw a square for this pixel (1x1 unit)
                # Invert Y because arrays go down, CAD goes up
                points = [
                    (c, -r),
                    (c + 1, -r),
                    (c + 1, -r - 1),
                    (c, -r - 1),
                    (c, -r)  # Close loop
                ]
                msp.add_lwpolyline(points, close=True)

    # Save to memory buffer
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode()


def generate_stl(grid_data, threshold=0.5, thickness=5):
    """Converts binary grid into 3D STL mesh (Voxel Extrusion)."""
    rows, cols = grid_data.shape
    faces = []

    # simple cube vertices template
    # 0,0,0 to 1,1,1

    for r in range(rows):
        for c in range(cols):
            if grid_data[r, c] > threshold:
                # Define 8 corners of the voxel cube
                x, y = c, rows - 1 - r  # Flip Y for 3D space

                # Create a cube at this position
                # Define 12 triangles (2 per face, 6 faces)
                # This is a simplified block generation for demonstration

                # vertices for a 1x1x(thickness) block
                p0 = [x, y, 0]
                p1 = [x+1, y, 0]
                p2 = [x+1, y+1, 0]
                p3 = [x, y+1, 0]
                p4 = [x, y, thickness]
                p5 = [x+1, y, thickness]
                p6 = [x+1, y+1, thickness]
                p7 = [x, y+1, thickness]

                # Define the 12 triangles (standard cube triangulation)
                # Bottom
                faces.append([p0, p2, p1])
                faces.append([p0, p3, p2])
                # Top
                faces.append([p4, p5, p6])
                faces.append([p4, p6, p7])
                # Front
                faces.append([p0, p1, p5])
                faces.append([p0, p5, p4])
                # Back
                faces.append([p2, p3, p7])
                faces.append([p2, p7, p6])
                # Left
                faces.append([p0, p4, p7])
                faces.append([p0, p7, p3])
                # Right
                faces.append([p1, p2, p6])
                faces.append([p1, p6, p5])

    # Create the mesh object
    np_faces = np.array(faces)
    surface = mesh.Mesh(np.zeros(np_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(np_faces):
        for j in range(3):
            surface.vectors[i][j] = f[j]

    # Save to memory buffer
    buffer = io.BytesIO()
    surface.save('temp.stl', fh=buffer)  # stl library needs a file-like object
    buffer.seek(0)
    return buffer.getvalue()


# --- 1. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Design Controls")

scenario = st.sidebar.selectbox(
    "Select Load Scenario",
    ("Custom (Manual)", "Cantilever Tip",
     "Mid-Span Load", "High-Shear Near Support")
)

if scenario == "Cantilever Tip":
    default_x, default_y = 58, 10
elif scenario == "Mid-Span Load":
    default_x, default_y = 30, 0
elif scenario == "High-Shear Near Support":
    default_x, default_y = 10, 19
else:
    default_x, default_y = 50, 10

st.sidebar.subheader("Boundary Conditions")
load_x = st.sidebar.slider("Load X Position (Length)", 0, 59, default_x)
load_y = st.sidebar.slider("Load Y Position (Height)", 0, 19, default_y)

st.sidebar.markdown("---")
st.sidebar.subheader("Manufacturing Constraints")
threshold = st.sidebar.slider(
    "Material Density Threshold", 0.0, 1.0, 0.35, 0.01)
st.sidebar.info(
    "Adjust threshold before exporting to ensure the part is solid.")

# --- 2. LOAD MODEL ---


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('neuro_topopt.keras')


try:
    model = load_model()
except:
    st.error("⚠️ Model not found. Please run 'train_stable.py' first.")
    st.stop()

# --- 3. INFERENCE ---
input_grid = np.zeros((20, 60))
r, c = load_y, load_x
r_min, r_max = max(0, r-1), min(19, r+1)
c_min, c_max = max(0, c-1), min(59, c+1)
input_grid[r_min:r_max+1, c_min:c_max+1] = 1.0

ai_input = input_grid.reshape(1, 20, 60, 1)
prediction = model.predict(ai_input, verbose=0).reshape(20, 60)

# --- 4. DASHBOARD ---
# Main view
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Setup")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.imshow(input_grid, cmap='gray_r')
    ax1.scatter([load_x], [load_y], c='#ff4b4b', s=100, label='Load')
    ax1.add_patch(plt.Rectangle((-1, 0), 1, 20, color='blue', alpha=0.5))
    ax1.axis('off')
    st.pyplot(fig1)

with col2:
    st.subheader("2. AI Prediction")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    # Overlay: Show heatmap but threshold it visually
    binary_view = (prediction > threshold).astype(float)
    ax2.imshow(1 - binary_view, cmap='gray')
    ax2.set_title(
        f"Optimized Structure (Volume: {np.mean(binary_view)*100:.1f}%)")
    ax2.axis('off')
    st.pyplot(fig2)

# --- 5. EXPORT SECTION (THE NEW UPGRADE) ---
st.markdown("---")
st.subheader("💾 Export for Manufacturing")
st.markdown("Download your AI-generated design as industry-standard CAD files.")

c1, c2, c3 = st.columns(3)

# Button 1: High-Res Image
buf = io.BytesIO()
fig2.savefig(buf, format="png", bbox_inches='tight', dpi=300)
buf.seek(0)
c1.download_button("📸 Download Blueprint (.PNG)",
                   buf, "blueprint.png", "image/png")

# Button 2: DXF (AutoCAD)
dxf_data = generate_dxf(prediction, threshold)
c2.download_button("📐 Download 2D CAD (.DXF)", dxf_data,
                   "design.dxf", "application/dxf")

# Button 3: STL (3D Printing)
stl_data = generate_stl(prediction, threshold, thickness=5)
c3.download_button("🧊 Download 3D Model (.STL)", stl_data,
                   "part.stl", "application/octet-stream")

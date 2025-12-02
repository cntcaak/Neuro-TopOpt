import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import os

# Import Core Modules
from core import manufacturing, postprocess, fem, analyzer, reporting, visualization

# --- PAGE CONFIG ---
st.set_page_config(page_title="Neuro-TopOpt Ultimate",
                   layout="wide", page_icon="🚀")
st.markdown(
    """<style>.stApp { background-color: #0e1117; color: white; }</style>""", unsafe_allow_html=True)

# --- HEADER IMAGE ---
# Use columns to center the image and constrain its size nicely.
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if os.path.exists("header.png"):
        st.image("header.png", use_container_width=True)

# Main Title appears below the image
st.title("🚀 Neuro-TopOpt: Enterprise Generative Design")
st.caption("Ultimate Edition | AI Design → Failure Prediction → AR Visualization")

# --- HOME PAGE THUMBNAIL (NEW) ---
# Use columns to center and constrain width to look like a thumbnail
t_col1, t_col2, t_col3 = st.columns([1, 2, 1])
with t_col2:
    if os.path.exists("app_thumbnail.png"):
        st.image("app_thumbnail.png", caption="Dashboard Preview",
                 use_container_width=True)

# --- SIDEBAR: SETUP ---
st.sidebar.header("🛠️ Design Setup")
scenario = st.sidebar.selectbox(
    "Load Case", ("Cantilever Tip", "Mid-Span", "Double Shear", "Custom"))

if scenario == "Cantilever Tip":
    dx, dy = 58, 10
elif scenario == "Mid-Span":
    dx, dy = 30, 0
elif scenario == "Double Shear":
    dx, dy = 58, 19
else:
    dx, dy = 50, 10

load_x = st.sidebar.slider("Load X", 0, 59, dx)
load_y = st.sidebar.slider("Load Y", 0, 19, dy)

st.sidebar.divider()
st.sidebar.header("🏭 Manufacturing")
mat_name = st.sidebar.selectbox("Material", list(postprocess.MATERIALS.keys()))
yield_strength = postprocess.MATERIALS[mat_name]['yield']
thresh = st.sidebar.slider("Material Threshold", 0.0, 1.0, 0.35)
smooth_fac = st.sidebar.slider("Smoothing Factor", 1, 5, 3)

# --- AI INFERENCE ---


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/neuro_topopt.keras')


try:
    model = load_model()
except:
    st.error("Model missing!")
    st.stop()

# Prepare Input
input_grid = np.zeros((20, 60))
r_min, r_max = max(0, load_y-1), min(19, load_y+1)
c_min, c_max = max(0, load_x-1), min(59, load_x+1)
input_grid[r_min:r_max+1, c_min:c_max+1] = 1.0

# Predict
pred = model.predict(input_grid.reshape(
    1, 20, 60, 1), verbose=0).reshape(20, 60)

# Process Results
smooth_pred = postprocess.smooth_grid(pred, factor=smooth_fac)
binary_pred = (smooth_pred > thresh).astype(float)
mass, cost = postprocess.calculate_cost_mass(binary_pred, mat_name)
mfg_report = analyzer.analyze_manufacturability(binary_pred)

# --- CACHED 3D GENERATION (Prevents Memory Crash) ---


@st.cache_data
def get_3d_plot(grid):
    return visualization.create_3d_voxel_plot(grid, thickness=10)


# --- TABS ---
t1, t2, t3, t4, t5, t6 = st.tabs(
    ["📐 Design", "🧠 AI Explainability", "🔥 Failure Mode", "🧊 3D Viewer", "📊 Report", "📥 Export"])

# TAB 1: DESIGN
with t1:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Boundary Conditions")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.imshow(1-input_grid, cmap='gray', vmin=0, vmax=1)
        ax.arrow(load_x, load_y - 3, 0, 2, head_width=2,
                 head_length=1, fc='red', ec='red')
        ax.scatter([load_x], [load_y], c='red', s=50, zorder=5)
        wall = plt.Rectangle((-1.5, -0.5), 1.5, 20.5,
                             color='blue', alpha=0.5, hatch='///')
        ax.add_patch(wall)
        ax.axis('off')
        ax.set_xlim(-5, 60)
        st.pyplot(fig)
        plt.close(fig)  # CLEANUP MEMORY
    with c2:
        st.subheader("AI Optimized Shape")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.imshow(1 - binary_pred, cmap='gray')
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close(fig2)  # CLEANUP MEMORY

# TAB 2: AI EXPLAINABILITY
with t2:
    st.subheader("🧠 Design Rationale")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(analyzer.explain_design(load_x, load_y, mass))
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3))
        im = ax.imshow(smooth_pred, cmap='magma')
        plt.colorbar(im, ax=ax, label="Importance")
        ax.set_title("Neural Attention Map")
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

# TAB 3: FAILURE MODE
with t3:
    st.subheader("🔥 Destructive Testing")
    test_load = st.slider("Test Load (N)", 100, 10000, 1000)

    if st.button("RUN SIMULATION"):
        with st.spinner("Simulating Physics..."):
            disp, stress = fem.run_fem_validation(
                pred > thresh, load_x, load_y, load_magnitude=test_load)

            if disp is not None:
                max_stress = np.max(stress)
                safety_factor = yield_strength / (max_stress + 1e-5)

                c1, c2, c3 = st.columns(3)
                c1.metric("Max Stress", f"{max_stress:.2e} Pa")
                c2.metric("Yield Strength", f"{yield_strength:.2e} Pa")

                if safety_factor < 1.0:
                    c3.metric(
                        "Safety Factor", f"{safety_factor:.2f}", delta="FAIL", delta_color="inverse")
                    st.error(f"🚨 FAILURE! Structure yielded at {test_load}N.")
                else:
                    c3.metric("Safety Factor",
                              f"{safety_factor:.2f}", delta="PASS")
                    st.success("✅ Structure Safe.")

                fig, ax = plt.subplots(figsize=(8, 3))
                masked_stress = np.ma.masked_where(pred <= thresh, stress)
                im = ax.imshow(masked_stress, cmap='jet', vmax=yield_strength)
                plt.colorbar(im, ax=ax, label="Stress (Pa)")
                ax.set_title("Stress Concentration (Red = Yield)")
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error("Simulation Failed")

# TAB 4: 3D VIEWER & AR
with t4:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("🧊 Interactive 3D Model")
        # Use Cached Plot Function
        fig_3d = get_3d_plot(binary_pred)
        st.plotly_chart(fig_3d, use_container_width=True)

    with c2:
        st.subheader("📱 AR Preview")
        try:
            qr_img = visualization.generate_ar_qrcode()
            st.image(qr_img.convert("RGB"),
                     caption="Scan for Mobile View", use_container_width=True)
        except:
            st.warning("Install 'qrcode': pip install qrcode pillow")

# TAB 5: REPORT
with t5:
    st.subheader("📋 Report Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mass", f"{mass:.3f} kg")
    c2.metric("Cost", f"${cost:.2f}")
    c3.metric("DfAM Score", f"{mfg_report['score']}")

# TAB 6: EXPORT
with t6:
    st.subheader("💾 Export Files")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### CAD Geometry")
        st.download_button("⬇️ Download DXF", manufacturing.generate_dxf(
            smooth_pred, thresh), "part.dxf", use_container_width=True)

    with c2:
        st.markdown("##### 3D Print File")
        st.download_button("⬇️ Download STL", manufacturing.generate_stl(
            smooth_pred, thresh), "part.stl", use_container_width=True)

    with c3:
        st.markdown("##### Report")
        metrics = {"Material": mat_name, "Mass": f"{mass:.3f}kg",
                   "Verdict": mfg_report['status']}
        pdf_bytes = reporting.generate_pdf(
            input_grid, binary_pred, metrics, load_x, load_y)
        st.download_button("⬇️ Download PDF", pdf_bytes, "report.pdf",
                           "application/pdf", use_container_width=True)

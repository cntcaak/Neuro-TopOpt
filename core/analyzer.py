import numpy as np
from scipy.ndimage import label, distance_transform_edt


def analyze_manufacturability(binary_grid):
    """Checks for printing violations."""
    results = {"score": 100, "islands": 0,
               "thin_walls": 0, "status": "Ready to Print"}

    # 1. Islands
    labeled_array, num_features = label(binary_grid)
    if num_features > 1:
        results["islands"] = num_features - 1
        results["score"] -= (results["islands"] * 10)

    # 2. Thin Walls
    thickness_map = distance_transform_edt(binary_grid)
    thin_mask = (thickness_map < 1.0) & (binary_grid > 0)
    thin_pixel_count = np.sum(thin_mask)
    if thin_pixel_count > 10:
        results["thin_walls"] = int(thin_pixel_count)
        results["score"] -= min(30, int(thin_pixel_count / 5))

    results["score"] = max(0, results["score"])

    if results["score"] == 100:
        results["status"] = "✅ Manufacturing Ready"
    elif results["score"] > 70:
        results["status"] = "⚠️ Minor Tooling Issues"
    else:
        results["status"] = "❌ Non-Manufacturable"

    return results


def explain_design_structured(load_x, load_y, mass_kg, vol_frac):
    """Returns structured insights for the dashboard."""
    insights = {"strategy": "", "physics": "", "efficiency": ""}

    # Strategy
    dist_x = abs(load_x - 0)
    if dist_x > 45:
        insights['strategy'] = "Long-Reach Cantilever: The AI prioritized a truss-web pattern to maximize stiffness over a long span, minimizing bending moment deflection."
    elif dist_x < 15:
        insights['strategy'] = "Short-Shear Block: The AI generated a solid shear block. Due to the proximity to the support, shear forces dominate over bending moments."
    else:
        insights[
            'strategy'] = "Balanced Beam: The design shows a hybrid topology, balancing material distribution between the tension (top) and compression (bottom) chords."

    # Physics
    if load_y < 8:
        insights['physics'] = "Top-Loading: You applied force to the upper fiber. The AI created a compressive strut to transfer this load diagonally down to the support."
    elif load_y > 12:
        insights[
            'physics'] = "Bottom-Hanging: You applied force to the bottom. The AI created a tension-tie (suspension cable style) to hold the load from the top support."
    else:
        insights['physics'] = "Neutral Axis Loading: The load is central. The AI distributed material symmetrically to handle pure bending stress."

    # Efficiency
    insights['efficiency'] = f"The topology utilized {vol_frac*100:.1f}% of the design domain ({mass_kg:.2f}kg). It successfully removed {(1-vol_frac)*100:.1f}% of inactive material that was not carrying significant stress."

    return insights

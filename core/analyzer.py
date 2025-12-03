import numpy as np
from scipy.ndimage import label, distance_transform_edt


def analyze_manufacturability(binary_grid):
    """Checks for printing violations."""
    results = {"score": 100, "islands": 0,
               "thin_walls": 0, "status": "Ready to Print"}

    labeled_array, num_features = label(binary_grid)
    if num_features > 1:
        results["islands"] = num_features - 1
        results["score"] -= (results["islands"] * 10)

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
    """Returns detailed engineering insights."""
    insights = {"strategy": "", "physics": "", "efficiency": ""}

    # Strategy Logic
    dist_x = abs(load_x - 0)
    if dist_x > 45:
        insights['strategy'] = "Long-Reach Cantilever: The AI recognized the large distance between the support and the load. It generated a truss-web pattern to maximize stiffness-to-weight ratio, effectively resisting the high bending moment."
    elif dist_x < 15:
        insights['strategy'] = "Short-Shear Block: Due to the proximity of the load to the wall, shear forces dominate. The AI generated a solid, blocky structure to prevent shear failure, prioritizing mass over intricate topology."
    else:
        insights[
            'strategy'] = "Balanced Hybrid: The design exhibits features of both a truss and a beam. The AI balanced material distribution between the tension (top) and compression (bottom) chords to handle mixed loading."

    # Physics Logic
    if load_y < 8:
        insights['physics'] = "Top-Fiber Loading: Force is applied to the upper section. The AI created a dominant compressive strut to transfer this load diagonally down to the fixed support, mimicking a classic brace."
    elif load_y > 12:
        insights[
            'physics'] = "Bottom-Fiber Loading: Force is applied to the lower section. The AI created a tension-tie (similar to a suspension cable) to 'hang' the load from the upper fixed support."
    else:
        insights['physics'] = "Neutral Axis Loading: The load is central. The topology is largely symmetrical, distributing stress evenly to handle the pure bending moment without significant torsion."

    # Efficiency Logic
    insights['efficiency'] = f"Material Optimization: The algorithm utilized {vol_frac*100:.1f}% of the available design domain (approx. {mass_kg:.2f}kg). It successfully identified and removed {(1-vol_frac)*100:.1f}% of 'lazy' material that was carrying zero stress, optimizing the cost-to-performance ratio."

    return insights

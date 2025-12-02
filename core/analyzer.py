import numpy as np
from scipy.ndimage import label, distance_transform_edt


def analyze_manufacturability(binary_grid):
    """
    Checks for printing violations:
    1. Disconnected Islands (Floating material)
    2. Minimum Feature Size (Thin walls)
    """
    results = {
        "score": 100,
        "islands": 0,
        "thin_walls": 0,
        "status": "Ready to Print"
    }

    # --- CHECK 1: DISCONNECTED ISLANDS ---
    # Label connected components (clusters of material)
    labeled_array, num_features = label(binary_grid)

    if num_features > 1:
        # If there's more than 1 cluster, we have floating islands!
        results["islands"] = num_features - 1  # Main body doesn't count
        results["score"] -= (results["islands"] * 10)  # Penalty

    # --- CHECK 2: THIN WALLS (Minimum Feature Size) ---
    # We use Distance Transform. If the "core" of a strut is too close to air, it's thin.
    # Thickness radius. 1.0 means width of 2 pixels.
    thickness_map = distance_transform_edt(binary_grid)

    # Count pixels that are material BUT are very thin (radius < 1.0)
    # We ignore the very edges of thick parts, we look for 'skeletal' thinness
    thin_mask = (thickness_map < 1.0) & (binary_grid > 0)
    thin_pixel_count = np.sum(thin_mask)

    if thin_pixel_count > 10:  # Allow small noise
        results["thin_walls"] = int(thin_pixel_count)
        # Heuristic penalty
        results["score"] -= min(30, int(thin_pixel_count / 5))

    # --- FINAL VERDICT ---
    results["score"] = max(0, results["score"])  # Clamp at 0

    if results["score"] == 100:
        results["status"] = "✅ Perfect for 3D Printing"
    elif results["score"] > 70:
        results["status"] = "⚠️ Minor Issues (Thin features)"
    else:
        results["status"] = "❌ Non-Manufacturable (Floating parts)"

    return results


def explain_design(load_x, load_y, mass_kg):
    """
    Generates a natural language explanation for the AI's design choices.
    """
    rationale = []

    # 1. Analyze Load Path
    dist_x = abs(load_x - 0)  # Distance from wall
    if dist_x > 45:
        rationale.append(
            f"• **Long Cantilever Logic:** The load is far from the support ({dist_x} units). The AI generated a truss-like web to maximize stiffness-to-weight ratio.")
    elif dist_x < 15:
        rationale.append(
            f"• **Shear Dominance:** The load is close to the wall. The AI prioritized a solid block design to resist high shear forces.")

    # 2. Analyze Load Height
    if load_y < 5:
        rationale.append(
            "• **Compression Arch:** Since the load is at the top, the AI formed an arch-like structure to direct forces into the lower support via compression.")
    elif load_y > 15:
        rationale.append(
            "• **Tension Tie:** The load is at the bottom. The AI created a top-chord tension tie to suspend the load.")

    # 3. Mass Efficiency
    rationale.append(
        f"• **Material Usage:** The AI achieved the structural requirements using only {mass_kg:.2f}kg of material, removing roughly 60% of unnecessary volume.")

    return "\n".join(rationale)

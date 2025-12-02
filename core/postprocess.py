import numpy as np
from scipy.ndimage import zoom

MATERIALS = {
    "Structural Steel (ASTM A36)": {"density": 7850, "price": 0.80, "color": "gray", "yield": 250e6},
    "Aluminum 6061-T6": {"density": 2700, "price": 2.50, "color": "silver", "yield": 276e6},
    "Titanium Ti-6Al-4V": {"density": 4430, "price": 40.00, "color": "purple", "yield": 880e6},
    "PLA Plastic (3D Print)": {"density": 1240, "price": 20.00, "color": "orange", "yield": 50e6}
}


def smooth_grid(grid, factor=3):
    return zoom(grid, factor, order=3)


def calculate_cost_mass(binary_grid, material_key, thickness_mm=10):
    props = MATERIALS[material_key]
    pixel_area_m2 = (0.01 * 0.01)
    thickness_m = thickness_mm / 1000.0
    filled_pixels = np.sum(binary_grid)
    volume_m3 = filled_pixels * pixel_area_m2 * thickness_m
    mass_kg = volume_m3 * props['density']
    cost = mass_kg * props['price']
    return mass_kg, cost

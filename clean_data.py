import os
import numpy as np

print("Starting Data Cleaning...")

load_dir = 'data/loads'
struct_dir = 'data/structures'

files = sorted(os.listdir(load_dir))
deleted_count = 0

for f in files:
    if f.endswith('.npy'):
        # Construct paths
        load_path = os.path.join(load_dir, f)
        struct_path = os.path.join(struct_dir, f)

        try:
            # Load arrays
            load_arr = np.load(load_path)
            struct_arr = np.load(struct_path)

            # CHECK FOR NaNs or INFINITY
            if np.isnan(load_arr).any() or np.isnan(struct_arr).any() or \
               np.isinf(load_arr).any() or np.isinf(struct_arr).any():

                print(f"Toxic Data found! Deleting {f}...")
                os.remove(load_path)
                os.remove(struct_path)
                deleted_count += 1

        except Exception as e:
            # If file is corrupted and can't be read, delete it too
            print(f"Corrupted file {f}: {e}. Deleting...")
            if os.path.exists(load_path):
                os.remove(load_path)
            if os.path.exists(struct_path):
                os.remove(struct_path)
            deleted_count += 1

print("------------------------------------------------")
print(f"Cleanup Complete. Deleted {deleted_count} toxic files.")
print(f"Remaining clean files: {len(os.listdir(load_dir))}")

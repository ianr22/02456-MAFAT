import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

# Sigma values for 12 blur levels
num_levels = 12
sigmas = np.linspace(0.0, 4.0, num_levels)

# Load TRUE mAP values from result text files
results_dir = '/home/junwoo/cmda4864_capstones/02456-MAFAT/results'
true_map = []

for blur_level in range(num_levels):
    # Try to find the result text file for this blur level
    pattern = os.path.join(results_dir, f'blur_{blur_level}_run_*', f'blur_{blur_level}_run_*.txt')
    matches = glob.glob(pattern)
    
    if not matches:
        # Try alternate pattern without run folder
        pattern = os.path.join(results_dir, f'blur_{blur_level}', f'blur_{blur_level}.txt')
        matches = glob.glob(pattern)
    
    if matches:
        txt_path = matches[0]
        try:
            with open(txt_path, 'r') as f:
                content = f.read()
            
            # Extract TRUE mAP value using regex
            match = re.search(r'TRUE mAP\s*=\s*([0-9.]+)', content)
            if match:
                map_value = float(match.group(1))
                true_map.append(map_value)
                print(f"blur_{blur_level}: TRUE mAP = {map_value}")
            else:
                print(f"Could not find TRUE mAP in blur_{blur_level}")
                true_map.append(0.0)
        except Exception as e:
            print(f"Error loading blur_{blur_level}: {e}")
            true_map.append(0.0)
    else:
        print(f"No result file found for blur_{blur_level}")
        true_map.append(0.0)

print(f"\nLoaded mAP values: {true_map}")

plt.figure(figsize=(9,5))
plt.plot(sigmas, true_map, marker='o', linestyle='-')
plt.title("TRUE mAP vs Gaussian Blur Sigma")
plt.xlabel("Sigma (Gaussian Blur)")
plt.ylabel("TRUE mAP")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()

plt.savefig('/home/junwoo/cmda4864_capstones/02456-MAFAT/results/true_map_plot.png', dpi=300)

plt.show()

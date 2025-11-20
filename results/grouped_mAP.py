import matplotlib.pyplot as plt
import numpy as np
import os



# Debug: print where file will be saved
print("Current working directory:", os.getcwd())

# Sigma values for 12 blur levels
num_levels = 12
sigmas = np.linspace(0.0, 4.0, num_levels)

# mAP values
general_map = [0.995777, 0.995713, 0.981622, 0.953830, 0.919275, 0.869777, 0.817436, 0.761823, 0.699140, 0.635385, 0.587840, 0.554515]
subclass_map = [0.922184, 0.920429, 0.854219, 0.757630, 0.670225, 0.534055, 0.423370, 0.324654, 0.243259, 0.191688, 0.149519, 0.117001]
color_map = [0.929539, 0.924856, 0.845041, 0.784158, 0.730761, 0.684211, 0.647124, 0.609569, 0.576648, 0.544188, 0.515970, 0.490671]
features_map = [0.898982, 0.892999, 0.764957, 0.628311, 0.508676, 0.394489, 0.316528, 0.258306, 0.168800, 0.093889, 0.061299, 0.046457]

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(sigmas, general_map, marker="o", label="General Class")
plt.plot(sigmas, subclass_map, marker="o", label="Sub-Class")
plt.plot(sigmas, color_map, marker="o", label="Color")
plt.plot(sigmas, features_map, marker="o", label="Features")

plt.xlabel("Sigma (Gaussian Blur)")
plt.ylabel("Grouped mAP")
plt.title("Grouped mAP vs Gaussian Blur Sigma")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save PNG next to script file
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "grouped_mAP_vs_blur.png")
plt.savefig(save_path, dpi=300)
print("Saved image to:", save_path)

plt.show()

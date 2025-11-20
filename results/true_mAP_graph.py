import matplotlib.pyplot as plt
import numpy as np

# Sigma values for 12 blur levels
num_levels = 12
sigmas = np.linspace(0.0, 4.0, num_levels)

# Corresponding TRUE mAP values
true_map = [
    0.920227, 0.916559, 0.830171, 0.732030, 0.644382, 0.539404,
    0.458399, 0.388370, 0.315836, 0.260169, 0.223833, 0.198565
]

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

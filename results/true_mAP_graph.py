import matplotlib.pyplot as plt

# Filenames and corresponding TRUE mAP values
file_labels = [
    "blur_0", "blur_1", "blur_2", "blur_3", "blur_4", "blur_5",
    "blur_6", "blur_7", "blur_8", "blur_9", "blur_10", "blur_11"
]
true_map = [
    0.920227, 0.916559, 0.830171, 0.732030, 0.644382, 0.539404,
    0.458399, 0.388370, 0.315836, 0.260169, 0.223833, 0.198565
]

plt.figure(figsize=(9,5))
plt.plot(file_labels, true_map, marker='o', linestyle='-')
plt.title("TRUE mAP of 12 Blur Runs")
plt.xlabel("Run File")
plt.ylabel("TRUE mAP")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()

plt.savefig('/home/junwoo/cmda4864_capstones/02456-MAFAT/results/true_map_plot.png', dpi=300)

plt.show()

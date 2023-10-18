import numpy as np
import matplotlib.pyplot as plt

X = np.load("emnist_hex_images.npy")
y = np.load("emnist_hex_labels.npy")

#Code for looking at the images:
for i in range(5):
    plt.imshow(X[i].reshape(20,20), cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")
    plt.show()

from collections import Counter
counter = Counter(y)
print(counter)
import numpy as np
import matplotlib.pyplot as plt
from unsupervised import SVD

# Load image
img = plt.imread("images/DanielV.jpg")

# Convert image to grayscale
img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# Apply SVD with increasing number of singular values
n_sv_values = [1, 5, 10, 25, 50, 100, 200, 300, 400, 500]

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))
axes = axes.flatten()

# Plot reconstructed images
for i, n_sv in enumerate(n_sv_values):
    # Apply SVD
    svd = SVD(n_components=n_sv)
    img_svd = svd.fit_transform(img_gray)
    img_reconstructed = svd.inverse_transform(img_svd)

    # Plot reconstructed image
    axes[i].imshow(img_reconstructed, cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(f"{n_sv} singular values")

# Plot original image
axes[-1].imshow(img_gray, cmap="gray")
axes[-1].axis("off")
axes[-1].set_title("Original Image")

plt.show()
import cv2
import matplotlib.pyplot as plt
from myImageGradient import myImageGradient

# Read the image in grayscale
img0 = cv2.imread("images/hawaii_hall.jpg", cv2.IMREAD_GRAYSCALE)

# Compute image gradients
imgx, imgy = myImageGradient(img0)

# Display original image and gradients
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img0, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(imgx, cmap="gray")
plt.title("Horizontal Gradient (Ix)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(imgy, cmap="gray")
plt.title("Vertical Gradient (Iy)")
plt.axis("off")

plt.tight_layout()
plt.savefig("images/image_gradients.png", dpi=300)
plt.show()
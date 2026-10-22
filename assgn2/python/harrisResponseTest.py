
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from myImageGradient import myImageGradient
from myHarrisResponse import myHarrisResponse

# Get the directory containing this script
script_dir = Path(__file__).resolve().parent

# Load the image in grayscale
image_path = script_dir /"images/hawaii_hall.jpg"
img0 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

# Check if image loaded successfully
if img0 is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Compute image gradients
imgx, imgy = myImageGradient(img0)

# Compute Harris response
R, Sxx, Syy, Sxy = myHarrisResponse(
    imgx, imgy, sigma=1.5, k=0.04
)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img0, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(Sxx, cmap="gray")
plt.title("Sxx")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(Syy, cmap="gray")
plt.title("Syy")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(Sxy, cmap="gray")
plt.title("Sxy")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(R, cmap="gray")
plt.title("Harris Response (R)")
plt.axis("off")

plt.tight_layout()

# Save figure for write-up
plt.savefig(script_dir / "images/harris_response.png", dpi=300)

plt.show()

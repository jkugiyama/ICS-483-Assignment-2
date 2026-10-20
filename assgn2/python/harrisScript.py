from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from myImageGradient import myImageGradient
from myHarrisResponse import myHarrisResponse
from myHarrisCorners import myHarrisCorners


def draw_corners(img, corners, radius=4):
    out = img.copy()
    for x, y in corners:
        cv2.circle(out, (int(x), int(y)), radius, (0, 0, 255), 1,
                   lineType=cv2.LINE_AA)
    return out


image_path = Path("images/hawaii_hall.jpg")
if not image_path.exists():
    raise FileNotFoundError(
        "images/hawaii_hall.jpg not found. Run: python downloadExampleImage.py"
    )

img = cv2.imread(str(image_path))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Parameters from the handout.
sigma = 1.5
k = 0.04
thresholdRel = 0.01
nmsSize = 7

imgx, imgy = myImageGradient(gray)
R, Sxx, Syy, Sxy = myHarrisResponse(imgx, imgy, sigma, k)
corners = myHarrisCorners(R, thresholdRel, nmsSize)

if any(v is None for v in [imgx, imgy, R, corners]):
    raise RuntimeError("Complete the TODOs in the Harris functions first.")

print(f"Detected corners: {len(corners)}")
result = draw_corners(img, corners)

# Intermediate results for the write-up.
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0, 0].set_title("Input")
ax[0, 1].imshow(imgx, cmap="gray")
ax[0, 1].set_title("Ix")
ax[1, 0].imshow(imgy, cmap="gray")
ax[1, 0].set_title("Iy")
ax[1, 1].imshow(R, cmap="gray")
ax[1, 1].set_title("Harris response R")
for a in ax.flat:
    a.axis("off")
plt.tight_layout()
plt.savefig("harris_intermediate.png", dpi=180, bbox_inches="tight")

plt.figure(figsize=(9, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f"Harris corners (N={len(corners)})")
plt.axis("off")
plt.tight_layout()
plt.savefig("harris_corners.png", dpi=180, bbox_inches="tight")
plt.show()

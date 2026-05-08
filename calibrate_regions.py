import cv2
import numpy as np
from PIL import Image

image = np.array(Image.open("training_images/Angel.jpeg"))
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
h, w = gray.shape[:2]

print(f"Image size: {w} x {h}")

regions = {
    # Body composition table (left) — shifted down
    "weight":          (0.17, 0.155, 0.10, 0.024),
    "body_fat":        (0.17, 0.180, 0.10, 0.024),
    "inorganic_salt":  (0.17, 0.205, 0.10, 0.024),
    "protein":         (0.17, 0.230, 0.10, 0.024),
    "body_water":      (0.17, 0.255, 0.10, 0.024),
    "muscle_mass":     (0.17, 0.280, 0.10, 0.024),
    "skeletal_muscle": (0.17, 0.305, 0.10, 0.024),

    # Obesity assessment (right) — repositioned
    "BMI":             (0.62, 0.360, 0.15, 0.025),
    "body_fat_rate":   (0.82, 0.415, 0.12, 0.025),

    # Other indicators (right) — repositioned
    "visceral_fat":    (0.88, 0.530, 0.10, 0.020),
    "BMR":             (0.84, 0.553, 0.14, 0.020),
    "fat_free_mass":   (0.84, 0.576, 0.14, 0.020),
    "subcutaneous_fat":(0.84, 0.599, 0.14, 0.020),
    "SMI":             (0.82, 0.622, 0.16, 0.020),
    "WHR":             (0.88, 0.668, 0.10, 0.020),
}

colors = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (128, 0, 255), (255, 128, 0), (0, 128, 255),
    (128, 255, 0), (0, 255, 128), (255, 0, 128),
    (64, 64, 255), (255, 64, 64), (64, 255, 64),
    (200, 100, 50),
]

for i, (name, (x_pct, y_pct, w_pct, h_pct)) in \
        enumerate(regions.items()):
    x1 = int(x_pct * w)
    y1 = int(y_pct * h)
    x2 = int((x_pct + w_pct) * w)
    y2 = int((y_pct + h_pct) * h)
    color = colors[i % len(colors)]

    # Draw rectangle
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    # Label with background
    label = name
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
    )
    cv2.rectangle(
        display,
        (x1, y1 - th - 6), (x1 + tw + 4, y1 - 2),
        color, -1
    )
    cv2.putText(
        display, label, (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
        (255, 255, 255), 1
    )

    # Also show the actual pixel coordinates
    print(f"  {name:20s}: "
          f"({x1:4d}, {y1:4d}) -> ({x2:4d}, {y2:4d})  |  "
          f"size: {x2-x1} x {y2-y1}")

cv2.imwrite("calibration_overlay.png", display)
print("\nSaved calibration_overlay.png")
print("Verify each rectangle covers ONLY the target number!")
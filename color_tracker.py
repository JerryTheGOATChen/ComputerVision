import cv2
import numpy as np

# HSV ranges for each colour, plus the BGR colour used to draw its box.
# Red needs two ranges because its hue wraps around 0/180.
COLORS = {
    "green": ([((40, 70, 40), (85, 255, 255))], (0, 255, 0)),
    "red":   ([((0, 100, 40), (10, 255, 255)), ((170, 100, 40), (180, 255, 255))], (0, 0, 255)),
    "blue":  ([((95, 50, 30), (130, 255, 255))], (255, 128, 0)),
}
MIN_AREA = 500  # ignore blobs smaller than this (in pixels)

cap = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
kernel = np.ones((5, 5), np.uint8)
print("Tracking green, red and blue. Click the Tracker window and press q to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from webcam")
        break
    frame = cv2.flip(frame, 1)  # mirror view so movement feels natural

    # 1. Smooth the image to reduce camera noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # 2. Convert to HSV and even out brightness to cope with lighting changes
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    hsv = cv2.merge([hue, sat, clahe.apply(val)])

    all_masks = np.zeros(hsv.shape[:2], np.uint8)
    row = 25
    for name, (ranges, box_color) in COLORS.items():
        # 3. Threshold: keep only pixels inside this colour's range
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        # 4. Clean the mask: opening removes specks, closing fills small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        all_masks |= mask

        # 5. Find the largest blob of this colour and mark it
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea) if contours else None
        if largest is not None and cv2.contourArea(largest) > MIN_AREA:
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(frame, f"{name} ({cx}, {cy})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            status = f"{name}: centre ({cx}, {cy})"
        else:
            status = f"{name}: not found"
        cv2.putText(frame, status, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        row += 25

    cv2.imshow("Tracker", frame)
    # Show only the pixels that passed the colour thresholds, in their real colours
    cv2.imshow("Mask", cv2.bitwise_and(blurred, blurred, mask=all_masks))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

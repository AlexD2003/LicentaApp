import cv2
import numpy as np

def full_preprocess(input_path, output_path, output_size=(240, 240)):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {input_path}")

    img = cv2.resize(img, (1024, 1024))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    h, w = mask.shape
    triangle = np.array([[0, 0], [int(w * 0.3), 0], [0, int(h * 0.3)]], np.int32)
    cv2.fillPoly(mask, [triangle], 0)

    cleaned = cv2.bitwise_and(img, img, mask=mask)

    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        cleaned = cleaned[y:y+bh, x:x+bw]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cleaned)
    resized = cv2.resize(enhanced, output_size)

    cv2.imwrite(output_path, resized)
    return output_path

import os
import cv2
import pydicom
import numpy as np

def save_and_convert(file, save_dir="static/uploads"):
    os.makedirs(save_dir, exist_ok=True)
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    path = os.path.join(save_dir, filename)
    file.save(path)

    if ext == ".dcm":
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.uint8)
        img = cv2.convertScaleAbs(img, alpha=255.0 / img.max())
        png_path = os.path.splitext(path)[0] + ".png"
        cv2.imwrite(png_path, img)
        return png_path
    else:
        return path

from pdf2image import convert_from_path
import cv2
import os
import numpy as np

for n in os.listdir("pdfs"):
    if os.path.splitext(n)[1] == ".pdf":
        for i, p in enumerate(convert_from_path(f"pdfs/{n}", grayscale=True)):
            cv2.imwrite(f"../datasets/all_ds/to_annotate/{n}_{i}.png", np.array(p))

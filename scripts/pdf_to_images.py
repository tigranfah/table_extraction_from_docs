from pdf2image import convert_from_path

import os
import cv2
import numpy as np

print("Reading and converting pdf pages to images...")
for ind, name in enumerate(os.listdir("../res/pdfs")):
    pdf_name = "../res/pdfs/" + name

    print(f"reading {name}.")
    pillow_page_images = convert_from_path(pdf_name, grayscale=True, poppler_path=r"C:\Users\user\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin")

    for i, im in enumerate(pillow_page_images):
        print(f"saving {name}_page_{i}.jpg")
        cv2.imwrite(f"../res/done/images/{name}_page_{i}.jpg", np.array(im))
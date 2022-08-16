import tensorflow as tf
import numpy as np
import cv2

import fitz

from infer_utils import detect_text_bboxes, detect_table_bboxes, normalize_table_detector_input, normalize_text_detector_input
from infer_utils import rescale_output, read_pdf_windowed
from infer_utils import TABLE_DETECTION_CONFIG, TEXT_DETECTION_CONFIG
from pdf2image import convert_from_path

import os

pdf_name = "sample.pdf"
fitz_doc = fitz.open(pdf_name)

print("Reading and converting pdf pages to images...")
pillow_page_images = convert_from_path(pdf_name, dpi=200, grayscale=True)

bpdf_name, ext = os.path.splitext(pdf_name)
if not os.path.exists(os.path.join("preds", bpdf_name)):
    os.mkdir(os.path.join("preds", bpdf_name))

print("Predicing...")
for page_i, pillow_image in enumerate(pillow_page_images):
    orig_img = np.array(pillow_image)
    normed_page_img = normalize_table_detector_input(orig_img, resize_shape=TABLE_DETECTION_CONFIG["input_shape"])
    table_bboxes = detect_table_bboxes(normed_page_img)

    rescaled_bboxes = rescale_output(table_bboxes, normed_page_img.shape[:2], orig_img.shape)

    for i, bbox in enumerate(rescaled_bboxes):
        table_img = orig_img[round(bbox[1]): round(bbox[3]), round(bbox[0]): round(bbox[2])]
        normed_table_img = normalize_text_detector_input(table_img, TEXT_DETECTION_CONFIG["input_shape"])
        # print(normed_talbe_img.shape)
        # print(normed_table_img.shape, table_img.shape)
        
        text_bboxes = detect_text_bboxes(normed_table_img)
        rescaled_text_bboxes = rescale_output(text_bboxes, normed_table_img.shape[1:][::-1], table_img.shape)
        # print(uint_normed.shape, type(uint_normed))
        # table_img = cv2.resize(table_img, normed_table_img.shape[1:])
        # print(table_img.shape, normed_table_img.shape)
        # text_list = []
        for box in rescaled_text_bboxes:
            cv2.rectangle(table_img, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), color=(0, 0, 255), thickness=1)
            box[1] = bbox[0] + box[1] - 5
            box[3] = bbox[1] + box[3] + 5
            box[0] = bbox[0] + box[0] - 5
            box[2] = bbox[1] + box[2] + 5
            text = read_pdf_windowed(fitz_doc[page_i], orig_img.shape, box)
            print(text, end="  ")
        print(f'Saving {os.path.join("preds", bpdf_name, f"page_{page_i+1}_table_{i+1}.png")}...')
        cv2.imwrite(os.path.join("preds", bpdf_name, f"page_{page_i+1}_table_{i+1}.png"), table_img)
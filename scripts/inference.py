import tensorflow as tf
import numpy as np
import cv2

import fitz

from infer_utils import detect_text_bboxes, detect_table_bboxes, normalize_table_detector_input, normalize_text_detector_input
from infer_utils import rescale_output, read_pdf_windowed, draw_table_struct, to_excel_file
from infer_utils import TABLE_DETECTION_CONFIG, TEXT_DETECTION_CONFIG
from pdf2image import convert_from_path

import os

pdf_name = "../res/pdfs/2207.12678.pdf"

fitz_doc = fitz.open(pdf_name)

print("Reading and converting pdf pages to images...")
pillow_page_images = convert_from_path(pdf_name, dpi=200, grayscale=True)

bpdf_name, ext = os.path.splitext(pdf_name.split("/")[-1])
if not os.path.exists(os.path.join("../res/preds", bpdf_name)):
    os.mkdir(os.path.join("../res/preds", bpdf_name))

if not os.path.exists(os.path.join("../res/excel", bpdf_name)):
    os.mkdir(os.path.join("../res/excel", bpdf_name))

if not os.path.exists(os.path.join("../res/masks", bpdf_name)):
    os.mkdir(os.path.join("../res/masks", bpdf_name))

read_indices = range(len(pillow_page_images))
# read_indices = [47]

print("Predicting...")
for page_i in read_indices:
    orig_img = np.array(pillow_page_images[page_i-1])
    normed_page_img = normalize_table_detector_input(orig_img, resize_shape=TABLE_DETECTION_CONFIG["input_shape"])
    table_bboxes = detect_table_bboxes(normed_page_img, bpdf_name, f"page-{page_i}")

    rescaled_bboxes = rescale_output(table_bboxes, normed_page_img.shape[:2], orig_img.shape)

    for box_i, table_bbox in enumerate(rescaled_bboxes):
        table_img = orig_img[round(table_bbox[1]): round(table_bbox[3]), round(table_bbox[0]): round(table_bbox[2])]
        normed_table_img = normalize_text_detector_input(table_img, TEXT_DETECTION_CONFIG["input_shape"])
        # print(normed_talbe_img.shape)
        # print(normed_table_img.shape, table_img.shape)
        
        text_bboxes = detect_text_bboxes(normed_table_img)
        rescaled_text_bboxes = rescale_output(text_bboxes, normed_table_img.shape[1:][::-1], table_img.shape)
        # print(fitz_doc[page_i-1].mediabox)
        draw_table_struct(table_img, rescaled_text_bboxes, bpdf_name, f"page-{page_i}_table-{box_i+1}")
        # print(rescaled_text_bboxes)
        for i in range(len(rescaled_text_bboxes)):
            rescaled_text_bboxes[i][0] += table_bbox[0]
            rescaled_text_bboxes[i][1] += table_bbox[1]
            rescaled_text_bboxes[i][2] += table_bbox[0]
            rescaled_text_bboxes[i][3] += table_bbox[1]

        rescaled_pdf_text_bboxes = rescale_output(rescaled_text_bboxes, orig_img.shape, (fitz_doc[page_i-1].mediabox[3], fitz_doc[page_i-1].mediabox[2]))
        # for b1, b2 in zip(rescaled_text_bboxes, rescaled_pdf_text_bboxes):
        #     print(b1, b2)
        # print(rescaled_pdf_text_bboxes, fitz_doc[page_i-1].mediabox, table_img.shape)

        to_excel_file(rescaled_pdf_text_bboxes, fitz_doc[page_i-1], bpdf_name, f"page-{page_i}_table-{box_i+1}")
        print(f"Page {page_i} - saved table {box_i+1}.")

        # draw_table_struct(table_img, rescaled_text_bboxes, bpdf_name, i)
        # text_list = []
        # print("page number", page_i-1)
        # for box in rescaled_text_bboxes:
        #     embrace = (3, 8)
        #     box[0] = table_bbox[0] + box[0] - embrace[0]
        #     box[1] = table_bbox[1] + box[1] - embrace[1]
        #     box[2] = table_bbox[0] + box[2] + embrace[0]
        #     box[3] = table_bbox[1] + box[3] + embrace[1]
        #     cv2.rectangle(orig_img, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), color=(0, 0, 255), thickness=1)
        #     # print("orig ", box)
        #     # print("modif ", box, "- orig shapes ", orig_img.shape, fitz_doc[page_i].mediabox)
        #     text = read_pdf_windowed(fitz_doc[page_i-1], orig_img.shape, box)
        #     print(text, end=" ")
        # print("\n")
        # print(f'Saving {os.path.join("preds", bpdf_name, f"page_{page_i+1}_table_{i+1}.png")}...')
        # cv2.imwrite(os.path.join("preds", bpdf_name, f"page_{page_i+1}_table_{i+1}.png"), orig_img)
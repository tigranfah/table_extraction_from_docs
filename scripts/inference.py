import tensorflow as tf
import numpy as np
import cv2

import fitz

with tf.device("CPU:0"):

    from infer_utils import detect_text_bboxes, detect_table_bboxes, normalize_table_detector_input, normalize_text_detector_input
    from infer_utils import rescale_output, rescale_output_cls, read_pdf_windowed, draw_table_struct, to_excel_file
    from infer_utils import TABLE_DETECTION_CONFIG, TEXT_DETECTION_CONFIG

    import os

    def pix2np(pix):
        im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        # im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    for name in os.listdir("../res/pdfs"):

        pdf_name = "../res/pdfs/" + name

        fitz_doc = fitz.open(pdf_name)

        # print("Reading and converting pdf pages to images...")
        dpi = 200
        dpi_matrix = fitz.Matrix(dpi / 72, dpi / 72)
        # pillow_page_images = convert_from_bytes(open(pdf_name,'rb').read(), grayscale=True, poppler_path=r"C:\Users\user\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin")
        pillow_page_images = []
        for page in fitz_doc:
            # print(dpi_matrix)
            # dpi_matrix = fitz.Matrix(page.mediabox[2], page.mediabox[3])
            # print("media", page.mediabox, dpi_matrix)
            page_pixel = page.get_pixmap(matrix=dpi_matrix)
            # page_pixel.set_dpi(dpi, dpi)
            pillow_page_images.append(pix2np(page_pixel))
            # page_pixel.save(f"{page.number}.png")
        # pillow_page_images = convert_from_path(pdf_name, grayscale=True, poppler_path=r"C:\Users\user\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin")

        bpdf_name, ext = os.path.splitext(pdf_name.split("/")[-1])
        if not os.path.exists(os.path.join("../res/preds", bpdf_name)):
            os.mkdir(os.path.join("../res/preds", bpdf_name))

        if not os.lpath.exists(os.path.join("../res/excel", bpdf_name)):
            os.mkdir(os.path.join("../res/excel", bpdf_name))

        if not os.path.exists(os.path.join("../res/masks", bpdf_name)):
            os.mkdir(os.path.join("../res/masks", bpdf_name))

        read_indices = range(1, len(pillow_page_images) + 1)
        # read_indices = [47]

        print("Extracting tables...")
        for page_i in read_indices:
            orig_img = pillow_page_images[page_i-1]
            # print(orig_img.min(), orig_img.max())
            normed_page_img = normalize_table_detector_input(orig_img, resize_shape=TABLE_DETECTION_CONFIG["input_shape"])
            table_bboxes = detect_table_bboxes(normed_page_img, bpdf_name, f"page-{page_i}")

            rescaled_bboxes = rescale_output(table_bboxes, normed_page_img.shape[:2], orig_img.shape)

            for box_i, table_bbox in enumerate(rescaled_bboxes):
                table_img = orig_img[round(table_bbox[1]): round(table_bbox[3]), round(table_bbox[0]): round(table_bbox[2])]
                normed_table_img = normalize_text_detector_input(table_img, TEXT_DETECTION_CONFIG["input_shape"])
                # print(normed_talbe_img.shape)
                # print(normed_table_img.shape, table_img.shape)
                
                rescaled_text_bboxes = detect_text_bboxes(normed_table_img)
                rescale_output_cls(rescaled_text_bboxes, normed_table_img.shape[1:][::-1], table_img.shape)
                # print(fitz_doc[page_i-1].mediabox)
                draw_table_struct(table_img, rescaled_text_bboxes, bpdf_name, f"page-{page_i}_table-{box_i+1}")

                for i in range(len(rescaled_text_bboxes)):
                    # rescaled_text_bboxes[i].min_x += table_bbox[0]
                    # rescaled_text_bboxes[i].min_y += table_bbox[1]
                    # rescaled_text_bboxes[i].max_x += table_bbox[0]
                    # rescaled_text_bboxes[i].max_y += table_bbox[1]
                    rescaled_text_bboxes[i].shift((table_bbox[0], table_bbox[1], table_bbox[0], table_bbox[1]))

                rescale_output_cls(rescaled_text_bboxes, orig_img.shape, (fitz_doc[page_i-1].mediabox[3], fitz_doc[page_i-1].mediabox[2]))
                # for b1, b2 in zip(rescaled_text_bboxes, rescaled_pdf_text_bboxes):
                #     print(b1, b2)
                # print(rescaled_pdf_text_bboxes, fitz_doc[page_i-1].mediabox, table_img.shape)

                to_excel_file(rescaled_text_bboxes, fitz_doc[page_i-1], bpdf_name, f"page-{page_i}_table-{box_i+1}")
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
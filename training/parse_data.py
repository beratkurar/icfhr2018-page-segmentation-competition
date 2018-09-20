import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import os


IMAGE_REGION_LABEL = 'ImageRegion'
GRAPHIC_REGION_LABEL = 'GraphicRegion'
TEXT_REGION_LABEL = 'TextRegion'
MAIN_TEXT_TYPE = 'paragraph'
MARGIN_TEXT_TYPE = 'marginalia'

IMAGE_REGION_VALUE = 128
MAIN_TEXT_REGION_VALUE = 0
MARGIN_TEXT_REGION_VALUE = 56
EMPTY_REGION_VALUE = 255

images_path = Path('./').glob('RASM2018_Example_Set/*.tif')

for path in images_path:
    image_path = str(path)
    print(image_path)

    xml_path = os.path.splitext(path)[0] + '.xml'

    img = cv2.imread(image_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page = root[1]

    number_of_regions = len(page)
    rec_main_text_regions = []
    rec_margin_text_regions = []
    rec_image_regions = []
    poly_main_text_regions = []
    poly_margin_text_regions = []
    poly_image_regions = []
    for i in range(0, number_of_regions):
        tag = page[i].tag

        if TEXT_REGION_LABEL not in tag and IMAGE_REGION_LABEL not in tag and GRAPHIC_REGION_LABEL not in tag:
            continue

        points = page[i][0].attrib.get('points').split(' ')
        number_of_vertices = len(points)
        vertices_list = []
        for j in range(number_of_vertices):
            x = int(points[j].split(',')[0])
            y = int(points[j].split(',')[1])
            vertices_list.append([x, y])
        if TEXT_REGION_LABEL in tag and page[i].attrib.get('type') == MAIN_TEXT_TYPE:
            if len(vertices_list) == 4:
                rec_main_text_regions.append(vertices_list)
            elif len(vertices_list) > 4:
                poly_main_text_regions.append(vertices_list)
        elif TEXT_REGION_LABEL in tag and page[i].attrib.get('type') == MARGIN_TEXT_TYPE:
            if len(vertices_list) == 4:
                rec_margin_text_regions.append(vertices_list)
            elif len(vertices_list) > 4:
                poly_margin_text_regions.append(vertices_list)
        elif IMAGE_REGION_LABEL in tag or GRAPHIC_REGION_LABEL in tag:
            if len(vertices_list) == 4:
                rec_image_regions.append(vertices_list)
            elif len(vertices_list) > 4:
                poly_image_regions.append(vertices_list)



    labeled = np.ones((img.shape[0], img.shape[1])) * EMPTY_REGION_VALUE

    main_text_pts = np.array(rec_main_text_regions, dtype=np.int32)
    margin_text_pts = np.array(rec_margin_text_regions, dtype=np.int32)
    images_pts = np.array(rec_image_regions, dtype=np.int32)

    labeled = cv2.fillPoly(labeled, images_pts, IMAGE_REGION_VALUE)
    labeled = cv2.fillPoly(labeled, main_text_pts, MAIN_TEXT_REGION_VALUE)
    labeled = cv2.fillPoly(labeled, margin_text_pts, MARGIN_TEXT_REGION_VALUE)

    for poly in poly_image_regions:
        labeled = cv2.fillPoly(labeled, np.array([poly], dtype=np.int32), IMAGE_REGION_VALUE)

    for poly in poly_main_text_regions:
        labeled = cv2.fillPoly(labeled, np.array([poly], dtype=np.int32), MAIN_TEXT_REGION_VALUE)

    for poly in poly_margin_text_regions:
        labeled = cv2.fillPoly(labeled, np.array([poly], dtype=np.int32), MARGIN_TEXT_REGION_VALUE)


    #cv2.imwrite('./RASM_labeled/Examples/'+os.path.basename(path), img)
    cv2.imwrite('ltrain/' + os.path.basename(path), labeled)


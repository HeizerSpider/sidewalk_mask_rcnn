from PIL import Image # (pip install Pillow)
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import json

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

if __name__ == '__main__':

    #cityscapes colour to ID
    color2id = {'(0, 0, 0)': 4, '(111, 74, 0)': 5, '(81, 0, 81)': 6, '(128, 64, 128)': 7, '(244, 35, 232)': 8, '(250, 170, 160)': 9, '(230, 150, 140)': 10, 
    '(70, 70, 70)': 11, '(102, 102, 156)': 12, '(190, 153, 153)': 13, '(180, 165, 180)': 14, '(150, 100, 100)': 15, '(150, 120, 90)': 16, '(153, 153, 153)': 18, 
    '(250, 170, 30)': 19, '(220, 220, 0)': 20, '(107, 142, 35)': 21, '(152, 251, 152)': 22, '(70, 130, 180)': 23, '(220, 20, 60)': 24, '(255, 0, 0)': 25, 
    '(0, 0, 142)': -1, '(0, 0, 70)': 27, '(0, 60, 100)': 28, '(0, 0, 90)': 29, '(0, 0, 110)': 30, '(0, 80, 100)': 31, '(0, 0, 230)': 32, '(119, 11, 32)': 33}

    test = Image.open('test.png')
    mask_images = [test]

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1
    is_crowd = 0

    # root = "/root/directory/here"

    # for image in root:
    #     image__ = Image.open(image)
    #     mask_images.append(image__)

    # Create the annotations
    annotations = []
    for mask_image in mask_images:
        sub_masks = create_sub_masks(mask_image)
        for color, sub_mask in sub_masks.items():
            category_id = color2id[color]
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            annotations.append(annotation)
            annotation_id += 1
        image_id += 1


    with open('data.json', 'w') as data:
        json.dump(annotations, data)
import cv2
import numpy as np


def divide_image(image_path):
    
    img = cv2.imread(image_path)

    rows, cols = img.shape[:2]
    cell_width = cols//3
    cell_height = rows//3

    cells = []

    for i in range (0,3):
        for j in range ( 0,3):
            cells.append(img[i*cell_height:i*cell_height+cell_height, j*cell_width:j*cell_width+cell_width])
    return cells

def colapse_image(cells):

    top  = np.concatenate((cells[0], cells[1], cells[2]), axis=1)
    mid  = np.concatenate((cells[3], cells[4], cells[5]), axis=1)
    bot  = np.concatenate((cells[6], cells[7], cells[8]), axis=1)
    full = np.concatenate((top, mid, bot), axis=0)

    return full



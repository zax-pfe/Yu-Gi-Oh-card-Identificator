import cv2
import numpy as np


# class image_prediction:

#     def __init__(self):
        

def return_resized_img_percent(img, percent):

    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized, width, height


def return_resized_img_size(img, height = 614 , width = 421):
    resized_image = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image





def reorder(myPoints):
 
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
 
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
 
    return myPointsNew

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

def extract_points(pts):
    points_list = []

    for i in pts:
        str1 = ' '.join(str(e) for e in i[0])
        split_list = str1.split(" ", 2)
        int_list = [int(i) for i in split_list]
        points_list.append(int_list)

    return points_list

def add_card_name(image, card_name, points_list):

    print("point list", points_list)
    cv2.putText(image, card_name, (points_list[0][0]+15,points_list[0][1]-15), cv2.FONT_HERSHEY_COMPLEX, 0.5 , (0,255,0), 2 )
    cv2.line(image, points_list[0],points_list[2], (0,255,0),thickness=3,lineType=2)
    cv2.line(image, points_list[2],points_list[3], (0,255,0),thickness=3,lineType=2)
    cv2.line(image, points_list[1],points_list[0], (0,255,0),thickness=3,lineType=2)
    cv2.line(image, points_list[3],points_list[1], (0,255,0),thickness=3,lineType=2)

    return image




def transform_for_contour_detection(resized):
    # add blur 
    imgBlur = cv2.GaussianBlur(resized, (7,7), 1) 

    # convert image into greyscale 
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    imgCanny = cv2.Canny(imgGray, 20, 255)

    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    return imgDil

def draw_circles_corner(img, points_list, color = (0,255,0), radius = 3, thickness = 3 ):
    # draw circle at the corner of the rectangle
    for point in points_list:
        x = point[0]
        y = point[1]
        cv2.circle(img,(x,y), radius, color, thickness)
        
    return img

def draw_corners_and_lines(image, screenCnt, lineColor = (0,255,0)):
        x0,y0 = screenCnt[0][0]
        x1,y1 = screenCnt[1][0]
        x2,y2 = screenCnt[2][0]
        x3,y3 = screenCnt[3][0]

        cv2.circle(image,(x0,y0), 3, lineColor, 2)
        cv2.circle(image,(x1,y1), 3, lineColor, 2)
        cv2.circle(image,(x2,y2), 3, lineColor, 2)
        cv2.circle(image,(x3,y3), 3, lineColor, 2)


        cv2.line(image, (x0,y0),(x1,y1), lineColor,thickness=2,lineType=2)
        cv2.line(image, (x1,y1),(x2,y2), lineColor,thickness=2,lineType=2)
        cv2.line(image, (x2,y2),(x3,y3), lineColor,thickness=2,lineType=2)
        cv2.line(image, (x3,y3),(x0,y0), lineColor,thickness=2,lineType=2)

        return image

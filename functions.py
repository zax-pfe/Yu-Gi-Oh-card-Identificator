import cv2
import numpy as np

def stack_images(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def return_resized_img_percent(img, percent):


    # height, width, channels = img.shape
    # print(f'Width: {width}, Height: {height}')



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

def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
 
    return img


def extract_points(pts):
    points_list = []

    for i in pts:
        str1 = ' '.join(str(e) for e in i[0])
        split_list = str1.split(" ", 2)
        int_list = [int(i) for i in split_list]
        points_list.append(int_list)

    return points_list


def getCountours(img, imgCountours):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 6000:
            # cv2.drawContours(imgCountours, cnt, -1, (255,0,255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y , w, h = cv2.boundingRect(approx)

            cv2.rectangle(imgCountours, (x,y), (x+w,y+h), (0,255,0), 5)

            cv2.putText(imgCountours, " card", (x + w + 10, y + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5 , (0,255,0), 2 )
            card = imgCountours[y:y + h, x:x + w]
            card_resized = return_resized_img_size(card)


    return card_resized, contours, imgCountours

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

import cv2
import numpy as np

# # Creation of the card detector class : this class uses open cv to detect the contours of the card and create a scanned version of the card

class Card_detection:
    """This class detect the yugioh card, this detection is made with opencv and line detection
    Once the border of the card is detected, it find the 4 corner of the card and transform the image
     like it was seen from above, this is what i called the "scanned card" """

    def __init__(self, color_text=(0,255,0), color_rectangle=(0,255,0)):
        self.color_rectangle = color_rectangle
        self.color_text = color_text

    def return_resized_img_percent(self,img, percent):
        """ This funciton resize the image to a choosen percentage
        - img : the image you want to resize
        - percent : percentage for the new size"""

        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized, width, height


    def return_resized_img_size(self,img, height = 614 , width = 421):
        """ This funciton resize the image to a choosen size
        - img : the image you want to resize
        - height and width : height and with of the resized image"""
        resized_image = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def reorder(self,myPoints):
    
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)
    
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] =myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] =myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
    
        return myPointsNew

    def biggestContour(self,contours):
        """ Take the list of all the contours and return the biggest contour of the list
        The biggest contour is suposed to be the yugioh card"""

        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 1000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest,max_area

    def extract_points(self,pts):
        """ take the points coordinates of the 4 corners and convert it into a list """
        points_list = []

        for i in pts:
            str1 = ' '.join(str(e) for e in i[0])
            split_list = str1.split(" ", 2)
            int_list = [int(i) for i in split_list]
            points_list.append(int_list)

        return points_list

    def add_card_name(self,image, card_name, points_list):
        """ take the image and return the image with the name and the border of the card drawn on it
        - image : the image in wich you want to print the name
        - card_name : the name predicted by the predictor 
        - points_list : Point list of the coodinates of the 4 corners of the card, this is used to draw the lines corresponding of the borders"""

        cv2.putText(image, card_name, (points_list[0][0]+15,points_list[0][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , self.color_text, 2 )
        cv2.line(image, points_list[0],points_list[2], self.color_rectangle,thickness=3,lineType=2)
        cv2.line(image, points_list[2],points_list[3], self.color_rectangle,thickness=3,lineType=2)
        cv2.line(image, points_list[1],points_list[0], self.color_rectangle,thickness=3,lineType=2)
        cv2.line(image, points_list[3],points_list[1], self.color_rectangle,thickness=3,lineType=2)

        return image

    def transform_for_contour_detection(self,img):
        """ transform the image for the contour detection : 
        - add blur
        - convert the image to grayscale
        - apply a Canny 
        - dilate """
        # add blur 
        imgBlur = cv2.GaussianBlur(img, (7,7), 1) 
        # convert image into greyscale 
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgGray, 20, 255)
        kernel = np.ones((5,5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

        return imgDil

    def return_scaned_card(self,path_image, image = None):
        """ this fucntion take an image path or an image and return the card seen from above, the scanned card
        - path_image : path of the image in wich we want to detect the card
        - image : image in wich we want to detect the card"""

        success = True

        if path_image=='null':
            img = image
        else:
            img = cv2.imread(path_image)
        #we first reduce the size of the image to 20%
        resized,width, height = self.return_resized_img_percent(img, 20)

        #this image will be the image plotted 
        self.card_plus_name, _, _=self.return_resized_img_percent(img, 40)

        height_original,width_original = img.shape[:2]
        

        """ Processing for card detection - resized """
        imgDil = self.transform_for_contour_detection(resized)
        # this function get the contours and create a pproximative bounding box
        contours, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # this fucntion is used to scan the card 
        biggest, _ = self.biggestContour(contours)
        imgWarpColored_original=None

        if biggest.size !=0:
            biggest=self.reorder(biggest)

            #get the size of the img and the 4 points at the corner to prepare the scan
            points = np.int32(biggest)
            pts1 = np.float32(biggest)
            pts1_original = pts1*5
            
            pts2 = np.float32([[0, 0],[width, 0], [0, height],[width, height]])
            pts2_original = pts2 * 5
    
            # get a list of the 4 points a the corner 
            self.point_list = self.extract_points(points)        

            matrix_original = cv2.getPerspectiveTransform(pts1_original, pts2_original)              
            imgWarpColored_original = cv2.warpPerspective(img, matrix_original, (width_original, height_original))


            return imgWarpColored_original, success

        else:
            success = False
            return _, success


    def card_prediction(self, card_name, setcode):
        """ Write the setcode and the name on the card"""

        point_for_card_plus_name = [[x * 2 for x in sublist] for sublist in self.point_list]
        self.card_plus_name = self.add_card_name(self.card_plus_name, card_name[0][0],point_for_card_plus_name)
        self.card_plus_name = cv2.putText(self.card_plus_name, setcode, (point_for_card_plus_name[0][0]+30,point_for_card_plus_name[0][1]-30), cv2.FONT_HERSHEY_COMPLEX, 0.5 , self.color_text, 2 )
        return self.card_plus_name

    def error_detection(self, path_image, error_type, image=None):
        """If an error occured during the detection of the image, we print error and the image"""

        if path_image == 'null':
            img = image
        else:
            img = cv2.imread(path_image)

        img, _, _=self.return_resized_img_percent(img, 40)
        height,width = img.shape[:2]

        # print('height,width :', height,width)
        if error_type == 1:
            cv2.putText(img, "error", (int(height/2),int(width/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , self.color_text, 2 )
        else:
            cv2.putText(img, "error detection", (int(height/2),int(width/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , self.color_text, 2 )
        return img






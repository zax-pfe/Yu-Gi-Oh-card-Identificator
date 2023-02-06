import cv2
import numpy as np


class Card_detection:

    def __init__(self, color_text=(0,255,0), color_rectangle=(0,255,0)):
        self.color_rectangle = color_rectangle
        self.color_text = color_text
        # print("card detector created")

    def return_resized_img_percent(self,img, percent):

        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # print("image resized to ",percent, " w :", width, " h : ", height)
        return resized, width, height


    def return_resized_img_size(self,img, height = 614 , width = 421):
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

    def extract_points(self,pts):
        points_list = []

        for i in pts:
            str1 = ' '.join(str(e) for e in i[0])
            split_list = str1.split(" ", 2)
            int_list = [int(i) for i in split_list]
            points_list.append(int_list)

        return points_list

    def add_card_name(self,image, card_name, points_list):

        cv2.putText(image, card_name, (points_list[0][0]+15,points_list[0][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , self.color_text, 2 )
        cv2.line(image, points_list[0],points_list[2], self.color_rectangle,thickness=3,lineType=2)
        cv2.line(image, points_list[2],points_list[3], self.color_rectangle,thickness=3,lineType=2)
        cv2.line(image, points_list[1],points_list[0], self.color_rectangle,thickness=3,lineType=2)
        cv2.line(image, points_list[3],points_list[1], self.color_rectangle,thickness=3,lineType=2)

        return image

    def transform_for_contour_detection(self,resized):
        # add blur 
        imgBlur = cv2.GaussianBlur(resized, (7,7), 1) 

        # convert image into greyscale 
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

        imgCanny = cv2.Canny(imgGray, 20, 255)

        kernel = np.ones((5,5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

        return imgDil

    def return_scaned_card(self,path_image):
        success = True
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

        # cv2.imwrite(path_image+'croped.jpg', imgWarpColored_original)



    def card_prediction(self, card_name, setcode):

        point_for_card_plus_name = [[x * 2 for x in sublist] for sublist in self.point_list]
        self.card_plus_name = self.add_card_name(self.card_plus_name, card_name[0][0],point_for_card_plus_name)
        self.card_plus_name = cv2.putText(self.card_plus_name, setcode, (point_for_card_plus_name[0][0]+30,point_for_card_plus_name[0][1]-30), cv2.FONT_HERSHEY_COMPLEX, 0.5 , self.color_text, 2 )


        cv2.imshow('card_plus_name', self.card_plus_name)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def error_detection(self, path_image, error_type):

        img = cv2.imread(path_image)
        img, _, _=self.return_resized_img_percent(img, 40)
        height,width = img.shape[:2]

        # print('height,width :', height,width)
        if error_type == 1:
            cv2.putText(img, "error", (int(height/2),int(width/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , self.color_text, 2 )
        else:
            cv2.putText(img, "error detection", (int(height/2),int(width/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , self.color_text, 2 )

        cv2.imshow('error', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()






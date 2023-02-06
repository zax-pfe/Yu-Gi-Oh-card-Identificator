# from paddleocr import PaddleOCR 
import cv2 

class Detect_setcode:


    def __init__(self,ocr_model):
        self.ocr_model = ocr_model

    def read_image(self, img_path):
        self.img = cv2.imread(img_path)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

    def crop_setcode(self):
        cropped_img = self.img[int(self.height*0.65):int(self.height*0.75), int(self.width*0.6):int(self.width*0.95)]
        return cropped_img

    def read_setcode(self, img):

        self.img=img
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

        image_cropped = self.crop_setcode()
        result = self.ocr_model.ocr(image_cropped)
        return result[0][0][1][0]

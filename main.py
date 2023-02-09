from core.detect_card import Card_detection
from core.predict_name import Predict
from keras import backend as K
from paddleocr import PaddleOCR 
import cv2
from core.set_code_detection import Detect_setcode
from core.identification import Card_identificator

ocr_model = PaddleOCR(lang='en')

#Parameters of the model
threshold = 0.5
interval = 0.2

#Path of the trained model and trained dictionnary:
model_trained_path = 'archive\\training_with_noise_28k.h5'
dict_trained_path = 'archive\\dict_28k.pkl'

#Creation of the class setcode detector, this class detect the setcode 
setcode_detector = Detect_setcode(ocr_model)

card_predictor = Predict(model_trained_path,dict_trained_path,threshold,interval)

card_detector = Card_detection()

card_identificator = Card_identificator(card_detector, card_predictor, setcode_detector)

image_path = 'images\WhatsApp Image 2023-01-13 at 10.19.22.jpeg'

image, name, setcode = card_identificator.identify_card(image_path)

cv2.imshow('full_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


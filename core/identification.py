from keras import backend as K
import core.transform_image as transform_image
from core.detect_card import Card_detection
from core.predict_name import Predict
from core.set_code_detection import Detect_setcode



class Card_identificator():

    def __init__(self, model_trained_path, dict_trained_path,threshold,interval) -> None:
        self.card_detector = Card_detection()
        self.predictor = Predict(model_trained_path,dict_trained_path,threshold,interval)
        self.setcode_detector = Detect_setcode()
    
    def identify_multiple_card(self, image_path):
        cells = transform_image.divide_image(image_path)
        cells_with_prediction = []
        card_names = []
        card_setcodes = []

        for cell in cells:
            try:
                scanned_card, success = self.card_detector.return_scaned_card(image=cell)
                if success==True:
                    card_name = self.predictor.predict_card_name(scanned_card)

                    try:
                        setcode = self.setcode_detector.read_setcode(scanned_card)
                    except:
                        setcode = 'setcode-error'

                    cell_with_prediction = self.card_detector.card_prediction(card_name, setcode, multiple=True) 
                else:
                    print("error during detection")
                    cell_with_prediction = self.card_detector.error_detection(path_image=None, error_type=0, image=cell, multiple=True)
                    card_name = 'error'
                    setcode = 'error'
            except Exception as e:
                print("error : ",e)
                card_name = 'error'
                setcode = 'error'
                cell_with_prediction = self.card_detector.error_detection( path_image=None,error_type=1, image=cell, multiple=True)

            card_names.append(card_name)
            card_setcodes.append(setcode)
            cells_with_prediction.append(cell_with_prediction)

        full_image = transform_image.colapse_image(cells_with_prediction)
        return full_image, card_names, card_setcodes

    def identify_card(self, path_image):

        try:
            scanned_card, success = self.card_detector.return_scaned_card(path_image)
            if success==True:
                card_name = self.predictor.predict_card_name(scanned_card)
                try:
                    setcode = self.setcode_detector.read_setcode(scanned_card)
                except:
                    setcode = 'setcode-error'

                card_with_prediction = self.card_detector.card_prediction(card_name, setcode)
                return card_with_prediction, card_name, setcode 
            else:
                print("error during detection")
                return self.card_detector.error_detection(path_image, error_type=0), "error", "error"

        except:
            print("error")
            return self.card_detector.error_detection(path_image, error_type=1), "error", "error"


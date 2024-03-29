import numpy as np
import keras
import pickle
import core.train as train
import core.transform_image as transform_image
from keras import backend as K

# #Creation of the class predictor : This class predict the card name using the artwork of the card 

class Predict:
    def __init__(self, model_trained_path, dict_trained_path, threshold, interval):

        self.triplet_loss = train.triplet_loss_t
        self.model = self.load_model(model_trained_path = model_trained_path)
        self.dict = self.load_dict(dict_trained_path = dict_trained_path)
        self.threshold = threshold
        self.interval = interval

    def load_model(self, model_trained_path):
        model=keras.models.load_model(model_trained_path,custom_objects={'triplet_loss_t':self.triplet_loss})
        print('model loaded successfully')
        return model

    def load_dict(self,dict_trained_path):
        with open(dict_trained_path, 'rb') as f:
            # Load the dictionary from the file
            loaded_dict = pickle.load(f)
        print("dictionary loaded successfully")
        return loaded_dict

    def encode_img(self, img1):
        img=img1[...,::-1]
        img=np.around(img/255,decimals=12)

        x_train=np.array([img])
        encoded=self.model.layers[3].predict_on_batch(x_train)
        return encoded

    def confidence_value(self,ref_encode,img_encode):
        dist=np.linalg.norm((img_encode-ref_encode))
        confidence=(self.threshold-max([dist,self.interval]))/(self.threshold-self.interval)
        return dist,confidence
        
    def compare_img_with_dict(self,image_to_test_path):
        """ This function take a path of an image an return a list of card names,
        the cards names are sorted by the more revelant to the less revelant 
        
        So the first name of the list should be the name of the given card image"""
        
        list_card_name = []
        img=transform_image.image_resizing(image_to_test_path)
        encoded_img = self.encode_img(img)
    
        for a in self.dict:
            image_dict_encoded = self.dict[a]
            dist,conf=self.confidence_value(encoded_img,image_dict_encoded)
            if dist<self.threshold:
                if conf > 0.0:
                    list_card_name.append((a,conf))
                    
        list_card_name = sorted(list_card_name, key=lambda x: x[1], reverse=True)
        return list_card_name

    def predict_card_name(self, image):
        """ This function take an image an return a list of card names,
    the cards names are sorted by the more revelant to the less revelant """
        
        list_card_name = []
        img=transform_image.image_resizing(path_image_to_test=None, test_on_dataset=False, image=image)
        encoded_img = self.encode_img(img)

        for a in self.dict:
            image_dict_encoded = self.dict[a]
            dist,conf=self.confidence_value(encoded_img,image_dict_encoded)
            if dist<self.threshold:
                if conf > 0.0:
                    list_card_name.append((a,conf))
                    
        list_card_name = sorted(list_card_name, key=lambda x: x[1], reverse=True)
        return list_card_name


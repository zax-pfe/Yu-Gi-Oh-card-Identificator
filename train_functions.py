#### Importing Libraries
from keras import backend as K
from keras.layers import *
import random
K.set_image_data_format('channels_first')
import glob
import random
import transform_image
import numpy as np
import model
from keras.models import Model
import pickle


def triplet_loss_t(self,y_true,y_pred):
    anchor=y_pred[:,0:128]
    pos=y_pred[:,128:256]
    neg=y_pred[:,256:384]

    positive_distance = K.sum(K.abs(anchor-pos), axis=1)
    negative_distance = K.sum(K.abs(anchor-neg), axis=1)
    probs=K.softmax([positive_distance,negative_distance],axis=0)
    loss=K.mean(K.abs(probs[0])+K.abs(1.0-probs[1]))
    return loss

class Train:
    def __init__(self,images_path, batch_size=32):
        self.model = model.FinalModel(input_shape=(3,244,244))
        self.list_images_path = self.get_list_images_path(images_path)
        self.batch_size = batch_size
        self.triplet_model = self.define_triplet_model()
    
    def get_list_images_path(self, images_path):
        list_images_path = glob.glob(images_path)
        print("the size of the dataset is :",len(list_images_path))
        return list_images_path

    def triplet_loss_t(self,y_true,y_pred):
        anchor=y_pred[:,0:128]
        pos=y_pred[:,128:256]
        neg=y_pred[:,256:384]

        positive_distance = K.sum(K.abs(anchor-pos), axis=1)
        negative_distance = K.sum(K.abs(anchor-neg), axis=1)
        probs=K.softmax([positive_distance,negative_distance],axis=0)
        loss=K.mean(K.abs(probs[0])+K.abs(1.0-probs[1]))
        return loss
        
    def return_random_path(self):
        dir = random.choice(self.list_images_path)
        return dir

    def define_triplet_model(self):

        triplet_model_a=Input((3,244,244))
        triplet_model_n=Input((3,244,244))
        triplet_model_p=Input((3,244,244))
        triplet_model_out=Concatenate()([self.model(triplet_model_a),self.model(triplet_model_p),self.model(triplet_model_n)])
        triplet_model=Model([triplet_model_a,triplet_model_p,triplet_model_n],triplet_model_out)

        print("---- triplet model created ----")

        return triplet_model

    def encode_img(self, img1):
        #img1=cv2.imread(path,1)
        img=img1[...,::-1]
        img=np.around(np.transpose(img,(2,0,1))/255,decimals=12)
        x_train=np.array([img])
        encoded=self.triplet_model.layers[3].predict_on_batch(x_train)
        return encoded


    def return_card_name(self, image_path):
        """ this function take the images path and return only the name of the card """

        card_folder = image_path.split('\\')[1]
        card_name = card_folder.split('-0-')[0]
        card_name = card_name.replace('-', ' ')
        return card_name
    
    def create_and_save_dict(self, name_of_the_dict = 'trained_dict.pkl'):
        """Creation of a dictionary containing the name of the card and the output of the model when we passed in the given card
        Creation of a list of tuples : (card name , tensor) in order to create then the dictionary """
        print("---- creation of the dictionary ....... ----")
        tuples_names_emb_list = []
        for i in (self.list_images_path):
            img=transform_image.image_resizing(i)
            tuples_names_emb_list.append((self.return_card_name(i), self.encode_img(img)))
        names_and_emb_dict = {key: value for (key, value) in tuples_names_emb_list}

        print('---- dictionary created successfully ----')

        """Now we save the dictionnay"""
        with open(name_of_the_dict, 'wb') as f:
            pickle.dump(names_and_emb_dict, f)
        
        print('---- dictionary saved successfully ----')

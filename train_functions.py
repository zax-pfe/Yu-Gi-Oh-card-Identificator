# YGO card Verification Using One Shot Learning

#### Importing Libraries


from keras import backend as K
from keras.layers import *
K.set_image_data_format('channels_first')
import random
import glob
import os
import random
import transform_image
import numpy as np


PATH='cardDatabaseFull'


"""# Defining the Triplet Loss Function"""

def triplet_loss_t(y_true,y_pred):
    print(y_pred)
    anchor=y_pred[:,0:128]
    pos=y_pred[:,128:256]
    neg=y_pred[:,256:384]
    
    positive_distance = K.sum(K.abs(anchor-pos), axis=1)
    negative_distance = K.sum(K.abs(anchor-neg), axis=1)
    probs=K.softmax([positive_distance,negative_distance],axis=0)
    #loss = positive_distance - negative_distance+alpha
    loss=K.mean(K.abs(probs[0])+K.abs(1.0-probs[1]))
    return loss


def data_gen(batch_size=32):
    """ The data generator will give to the model 3 images, this first one is the anchor:
    so its a random image choosen in the dataset, the second is the positive:
    its the first image but with transformation and the third one is the negative: its a different picture in the dataset"""
    while True:
        i=0
        positive=[]
        anchor=[]
        negative=[]    
        

        while(i<batch_size):
            r1=random.choice(os.listdir(PATH))
            r2=random.choice(os.listdir(PATH))

            p1=PATH+'/'+ r1
            id1=os.listdir(p1)
            dir1 = p1+'/'+ id1[0]

            p2=PATH+'/'+ r2
            id2=os.listdir(p2)
            dir2 = p2+'/'+ id2[0]

            pos_img = transform_image.augment_img_iaa(dir1,True)
            anc_img = transform_image.augment_img_iaa(dir1, False)
            # Not applying augmentation to the negative image by setting the second parameter to False
            neg_img = transform_image.augment_img_iaa(dir2,False)

            positive.append(list(pos_img))

            negative.append(list(neg_img))

            anchor.append(list(anc_img))

            i=i+1
        #return anchor,positive,negative
        yield ([np.array(anchor),np.array(positive),np.array(negative)],np.zeros((batch_size,1)).astype("float32"))
        
def return_random_path(images_path):
    dir = random.choice(images_path)
    return dir
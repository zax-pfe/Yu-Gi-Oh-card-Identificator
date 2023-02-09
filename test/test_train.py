# YGO card Verification Using One Shot Learning

#### Importing Libraries
from core.train import Train
import core.transform_image as transform_image
import numpy as np


images_path='cardDatabaseFull/*/*.jpg'
batch_size = 8

model_to_train = Train(images_path, batch_size=batch_size)

def data_gen():
    """ The data generator will give to the model 3 images, this first one is the anchor:
    so its a random image choosen in the dataset, the second is the positive:
    its the first image but with transformation and the third one is the negative: its a different picture in the dataset"""
    while True:
        i=0
        positive=[]
        anchor=[]
        negative=[]    

        while(i<model_to_train.batch_size):

            dir1 = model_to_train.return_random_path()
            dir2 = model_to_train.return_random_path()

            pos_img = transform_image.augment_img_iaa(dir1,True)
            anc_img = transform_image.augment_img_iaa(dir1, False)
            # Not applying augmentation to the negative image by setting the second parameter to False
            neg_img = transform_image.augment_img_iaa(dir2,False)

            positive.append(list(pos_img))

            negative.append(list(neg_img))

            anchor.append(list(anc_img))

            i=i+1
        #return anchor,positive,negative
        yield ([np.array(anchor),np.array(positive),np.array(negative)],np.zeros((model_to_train.batch_size,1)).astype("float32"))


model_to_train.triplet_model.compile(optimizer='adam',loss=model_to_train.triplet_loss_t)
model_to_train.triplet_model.fit( data_gen() , steps_per_epoch=10, epochs=2)
print('---- model saved successfully ----')
model_to_train.triplet_model.save('triplet_model_b_and_w_img244_epoch_15.h5')
print('---- model trained successfully ----')

model_to_train.create_and_save_dict()

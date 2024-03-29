#Import libraries needed
from keras import backend as K
import cv2
from core.identification import Card_identificator

#Parameters of the model
threshold = 0.5
interval = 0.2

#Path of the trained model and trained dictionnary:
model_trained_path = 'archive\\training_with_noise_28k.h5'
dict_trained_path = 'archive\\dict_28k.pkl'

#Creation of the Card identificator class : this class mix the three previous class to detect the card, predict the name and detect the setcode 
card_identificator = Card_identificator(model_trained_path,dict_trained_path,threshold,interval)

#Image path of the card you want to detect
image_path = 'images\WhatsApp Image 2023-01-13 at 10.19.22.jpeg'

#get image with anotation, the name and the setcode of a given image 
image, name, setcode = card_identificator.identify_card(image_path)


print(image, name)

#plot this image 
cv2.imshow('full_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


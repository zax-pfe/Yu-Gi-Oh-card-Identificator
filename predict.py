# YGO card Verification Using One Shot Learning

#### Importing Libraries
from predict_functions import Predict
from keras import backend as K


model_trained_path = 'archive\\triplet_model_b_and_w_img244_epoch_5.h5'
dict_trained_path = 'archive\\dict_black_white_244_padded.pkl'
threshold = 0.5
interval = 0.2

predictor = Predict(model_trained_path,dict_trained_path,threshold,interval)

card_path_to_test = 'image_croped_2\WhatsApp Image 2023-01-13 at 10.19.22.jpeg_croped.jpg'

"""# Compare the output of the model with the dictionnary"""

"""#### Test of the model """

card_name = predictor.compare_img_with_dict(card_path_to_test)
print(card_name[0])

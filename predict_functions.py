import numpy as np
import transform_image

"""## Function to convert the image to embeddings.

>It normalizes the image matrix and transpose it since here we are using the 'Channels First' data format. Then it uses the base model to predict the embedding for the provided image
"""

def encode_img(img1,triplet_model):
    #img1=cv2.imread(path,1)
    img=img1[...,::-1]
    img=np.around(np.transpose(img,(2,0,1))/255,decimals=12)
    x_train=np.array([img])
    emb=triplet_model.layers[3].predict_on_batch(x_train)
    return emb

"""## Function to calculate the distance between the embeddings and confidence score

>  Selecting the threshold value as 0.5 and intervals means that +/- 0.1 the model confidence score will be 100%.
"""
def confidence_value(ref_encode,img_encode,threshold=0.50,interval=0.1 ):
    #diff=np.max(img_encode-ref_encode)
    dist=np.linalg.norm((img_encode-ref_encode))
    #confidence=(1-K.eval(tf.minimum(dist,1)))
    confidence=(threshold-max([dist,interval]))/(threshold-interval)
    return dist,confidence


def compare_img_with_dict(image_to_test_path, names_and_emb_dict, triplet_model, threshold=0.5):
    """ This function take a path of an image an return a list of card names,
    the cards names are sorted by the more revelant to the less revelant 
    
    So the first name of the list should be the name of the given card image"""
    
    list_card_name = []
    img=transform_image.image_resizing(image_to_test_path)
    encoded_img = encode_img(img,triplet_model)
  
    for a in names_and_emb_dict:
        image_dict_encoded = names_and_emb_dict[a]
        dist,conf=confidence_value(encoded_img,image_dict_encoded)
        if dist<threshold:
            if conf > 0.0:
                list_card_name.append((a,conf))
                
    list_card_name = sorted(list_card_name, key=lambda x: x[1], reverse=True)
    return list_card_name



def compare(path_image, triplet_model, path_image2 = None,threshold=0.5):
    """ plot 2 images side to side and compute if the images math and with wich confidence"""
    
    if not path_image2:
        rimg=transform_image.image_resizing(path_image, True)
        img=transform_image.image_resizing(path_image)
    else :
        rimg=transform_image.image_resizing(path_image)
        img=transform_image.image_resizing(path_image2) 


    r_encode=encode_img(rimg,triplet_model)
    img_encode=encode_img(img,triplet_model)
    dist,conf=confidence_value(r_encode,img_encode)
    if dist<threshold:
        print("Match with a confidence of ",conf*100)
        #print("Distance ",dist)
    else:
        print("No Match with a confidence of ",abs(conf*100))

    return None
import numpy as np
import cv2
import imgaug.augmenters as iaa
import random

def padding_images(image):
    """ this function will randomly pad the image like it was the border of an image of the ygo card."""
    # generate a random value to select  the padding mode 
    padding_mode = random.randint(0, 8)	
    # generate a random value to generate the number of pixel padded
    padding_value1 = random.randint(0, 15)	
    # second random value for double transformation ( ex top left or bot right)
    padding_value2 = random.randint(0, 15)	
    #color of the padding 
    color_pad = (70,90)

    # top
    if padding_mode == 0:
        aug = iaa.CropAndPad(px=((padding_value1),(0),(-padding_value1),(0)),pad_cval=color_pad)
        padded = aug.augment_image(image)
        return padded

    # bot
    if padding_mode == 1:
        aug = iaa.CropAndPad(px=((-padding_value1),(0),(padding_value1),(0)),pad_cval=color_pad)
        padded = aug.augment_image(image)
        return padded

    # left
    if padding_mode == 2:
        aug = iaa.CropAndPad(px=((0),(-padding_value1),(0),(padding_value1)),pad_cval=color_pad)
        padded = aug.augment_image(image)
        return padded

    # right
    if padding_mode == 3:
        aug = iaa.CropAndPad(px=((0),(padding_value1),(0),(-padding_value1)),pad_cval=color_pad)
        padded = aug.augment_image(image)
        return padded

    # top_left
    if padding_mode == 4:
        aug = iaa.CropAndPad(px=((padding_value1),(-padding_value2),(-padding_value1),(padding_value2)),pad_cval=color_pad)
        padded = aug.augment_image(image)
        return padded

    # top_right
    if padding_mode == 5:
        aug = iaa.CropAndPad(px=((padding_value1),(padding_value2),(-padding_value1),(-padding_value2)),pad_cval=color_pad)
        padded = aug.augment_image(image)
        return padded

    # bot_left
    if padding_mode == 6:
        aug = iaa.CropAndPad(px=((-padding_value1),(-padding_value2),(padding_value1),(padding_value2)),pad_cval=color_pad)
        padded = aug.augment_image(image)
        return padded

    # bot_right
    if padding_mode == 7:
        aug = iaa.CropAndPad(px=((-padding_value1),(padding_value2),(padding_value1),(-padding_value2)),pad_cval=color_pad)
        padded = aug.augment_image(image)
        return padded

    # None
    else:
        return image
    
augmentation = iaa.Sequential([
    iaa.Resize({"height": 614, "width": 421}),
    iaa.Crop(px=(112, 52, 179, 52),), #top, right, bot, left
    iaa.Sometimes(0.75, iaa.GaussianBlur(sigma=(0, 3.0))),
    # iaa.Sometimes(0.25,iaa.OneOf([iaa.Dropout(p=(0, 0.1)),iaa.CoarseDropout(0.1, size_percent=0.1)])),
    iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
    iaa.Multiply((0.7,1.3)),
    iaa.LinearContrast((0.7,1.3)),
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
    iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5)),
    iaa.GammaContrast((0.5, 1.0), per_channel=True),
    iaa.Resize({"height": 244, "width": 244}),
    iaa.Grayscale(alpha=1.0)
])

augmentation_anchor = iaa.Sequential([
    iaa.Resize({"height": 614, "width": 421}),
#     iaa.WithBrightnessChannels(iaa.Add(10)),
    iaa.Crop(px=(112, 52, 179, 52),), #top, right, bot, left
    iaa.Resize({"height": 244, "width": 244}),
    iaa.Grayscale(alpha=1.0)

])
    
    

def augment_img_iaa(path_img, positive, transpose_and_normalize = True):
    """ This function will apply random transformation on an image
    - The first argument is the path of the image
    - The second argument is to say if the image we gonna transform is the positive image
    - transpose and normalize is used to transform image to feed the nn
    
    The transformation applyed are randomly : increse or deacrease brightness, add saturation,
    add contrast and crop and resize the card to keep only the image of the monster."""
    
    augmented_images = cv2.imread(path_img)
    temp_list_4aug = [augmented_images]

    if positive:

        augmented_images  = augmentation(images = temp_list_4aug )
        augmented_images = augmented_images[0]
        augmented_images = padding_images(augmented_images)
        
    else : 
        augmented_images = augmentation_anchor(images = temp_list_4aug )
        augmented_images = augmented_images[0]

    if transpose_and_normalize:
        # augmented_images=np.transpose(augmented_images,(2,0,1))
        augmented_images=augmented_images.astype('float32')/255.0


    return augmented_images

#To loacalize the mignature and resize the image
def image_resizing(path_image_to_test, test_on_dataset = False):
    image = augment_img_iaa(path_image_to_test, test_on_dataset, False)
    return image


num_classes = 3
from dataset_vgg16 import *
from dataset_resnet50 import *
from dataset_inception_v3 import *
from dataset_mobilenet_v3_small import *
from dataset_mobilenet_v3_large import *
 
IMAGES_VGG16 = [IMAGES_CLASS_VGG16_0, IMAGES_CLASS_VGG16_1, IMAGES_CLASS_VGG16_2]
IMAGES_INCEPTION = [IMAGES_CLASS_VGG16_0, IMAGES_CLASS_VGG16_1, IMAGES_CLASS_VGG16_2]
IMAGES_MNET_SMALL = [IMAGES_CLASS_VGG16_0, IMAGES_CLASS_VGG16_1, IMAGES_CLASS_VGG16_2]
IMAGES_MNET_LARGE = [IMAGES_CLASS_VGG16_0, IMAGES_CLASS_VGG16_1, IMAGES_CLASS_VGG16_2]
IMAGES_RESNET = [IMAGES_CLASS_VGG16_0, IMAGES_CLASS_VGG16_1, IMAGES_CLASS_VGG16_2]

def get_image_dataset(MODEL_NAME):
    if MODEL_NAME == 'vgg16':
        return IMAGES_VGG16
    elif MODEL_NAME == 'inception_v3':
        return IMAGES_INCEPTION_V3
    elif MODEL_NAME == 'mobilenet_v3_small':
        return IMAGES_MOBILENET_V3_SMALL
    elif MODEL_NAME == 'mobilenet_v3_large':
        return IMAGES_MOBILENET_V3_LARGE
    elif MODEL_NAME == 'resnet50':
        return IMAGES_RESNET50
    else :
        raise ValueError(f"Unknown model: {MODEL_NAME}. Supported models are: vgg16, inception_v3, mobilenet_v3_small, mobilenet_v3_large, resnet50.")

def get_layer_list(MODEL_NAME):
    if (MODEL_NAME == 'inception_v3'):
        return INCEPTION_V3_LAYERS
        

def get_lambda_val(MODEL_NAME):
    if (MODEL_NAME == 'inception_v3'):
        return lambda_inception
        
        

def get_model_path(MODEL_NAME, layer_name, lambda_val):
    base_path = '/mnt/data/results/' +MODEL_NAME.strip() + '/loss_' + MODEL_NAME.strip() + '_' + layer_name.strip() + '_' + str(lambda_val) + '.pth' 
    print(base_path)
    return (base_path)
    
    
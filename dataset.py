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
    if(MODEL_NAME == 'vgg16'):
        return 'VGG16_LAYERS'
    if(MODEL_NAME == 'resnet50'):
        return 'RESNET50_LAYERS'
    if(MODEL_NAME == 'mobilenet_v3_small'):
        return 'MOBILENET_V3_SMALL_LAYERS'
    if(MODEL_NAME == 'mobilenet_v3_large'):
        return 'MOBILENET_V3_LARGE_LAYERS'
        

def get_lambda_val(MODEL_NAME):
    if (MODEL_NAME == 'inception_v3'):
        return lambda_inception
    if(MODEL_NAME == 'vgg16'):
        return lambda_vgg16
    if(MODEL_NAME == 'resnet50'):
        return lambda_resnet50
    if(MODEL_NAME == 'mobilenet_v3_small'):
        return lambda_mobilenet_v3_small
    if(MODEL_NAME == 'mobilenet_v3_large'):
        return lambda_mobilenet_v3_large



def get_model_path(MODEL_NAME, layer_name, lambda_val):
    
    base_path = '/mnt/data/results/' +MODEL_NAME.strip() + '/loss_' + MODEL_NAME.strip() + '_' + layer_name.strip() + '_' + str(lambda_val) + '.pth' 
    print(base_path)
    try:
        with open(base_path, 'r') as f:
            pass
    except Exception as e:
        print("File not found in the given path ")
        raise FileNotFoundError(f"The file '{filepath}' was not found.")
    return (base_path)
    
    
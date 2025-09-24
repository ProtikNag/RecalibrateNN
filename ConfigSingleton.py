import argparse
import yaml
import os
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class ConfigSingleton:
    _instance = None

    def __new__(cls, config_file):
        if cls._instance is None:
            cls._instance = super(ConfigSingleton, cls).__new__(cls)
            cls._instance.config_file = config_file
            cls._instance._load_and_process_config()
           
        return cls._instance
        
    def is_subset(self, list1, list2):
      """
      Checks if list1 is a subset of list2.
      """
      return set(list1).issubset(set(list2))
    

    def _load_and_process_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                self.config = config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_file}")
        #REad all the misc sections
        self.SEED = config['misc']['seed']
        
        #Read all the classification sections
        self.CLASSIFICATION_DATA_BASE_PATH = config['classification']['data_base_path']
        self.TARGET_CLASS_LIST = config['classification']['target_class_list']

        self.LINEAR_CLASSIFIER_TYPE = config['classification']['linear_classifier_type']
        self.NUM_CLASSES = self._get_num_classes(self.CLASSIFICATION_DATA_BASE_PATH)
        self.LAMBDA_ALIGNS = config['classification']['lambda_aligns']
            

        #Read all the concept and random sections
        config_base_path = config['concept']['base_path']
        target_folder_path = config['concept']['target_folders']
        self.CONCEPT_FOLDER_LIST = [os.path.join(config_base_path, concept_name) for concept_name in target_folder_path]

        random_folder_path = config['concept']['random_folder']
        self.RANDOM_FOLDER = config['concept']['random_folder']

        #Read all the hyperparameters sections        
        self.LEARNING_RATE = float(config['hyperparameters']['learning_rate'])
        self.EPOCHS = int(config['hyperparameters']['epochs'])
        self.BATCH_SIZE = config['hyperparameters']['batch_size']
        self.DEVICE = DEVICE
        self._verify_all_paths(self.config)
        self._instance._override_recalibration(self.config)
        self._instance._read_xai_image_path(self.config)

        
    def _override_recalibration(self, config):
        #Recalibration 
        self.OVERRIDE_RECALIB           = config['recalibration']['override_retraining']
        self.VGG_RECALIB                = config['recalibration']['layers_to_train']['vgg16']
        self.RESNET50_RECALIB           = config['recalibration']['layers_to_train']['resnet50']
        self.INCEPTION_V3_RECALIB       = config['recalibration']['layers_to_train']['inception_v3'] 
        self.MOBILENET_V3_SMALL_RECALIB = config['recalibration']['layers_to_train']['mobilnet_v3_small'] 
        self.MOBILENET_V3_LARGE_RECALIB = config['recalibration']['layers_to_train']['mobilenet_v3_large'] 
        self.LAMBDA_ALIGNS_RECALIB = config['recalibration']['lambda_aligns_recalib']
        if(self.is_subset(self.LAMBDA_ALIGNS_RECALIB,self.LAMBDA_ALIGNS)):
            print(" recalibration lambdas is subset of the lambda aligns ")
        else:
            raise("Exception in lambda aligns and sensiticvity analysis ")
        
        if(self.OVERRIDE_RECALIB):
            print(" Overriding the recalibration layers and lambda aligns")
            self.LAMBDA_ALIGNS = self.LAMBDA_ALIGNS_RECALIB
        return True
    
    def _read_xai_image_path(self, config):
        self.INTEGRATED_GRADIENT = config['xai_before_after']['integrated_gradients']
        self.GRADCAM = config['xai_before_after']['grad_cam']
        self.LIME = config['xai_before_after']['lime']
        #Read all the xai image paths
        self.XAI_IMAGE_PATH = config['xai_before_after']['image_path']
        for i in self.XAI_IMAGE_PATH:
            if(os.path.isfile(i)):
                print(f"XAI image path {i} exists.")
            elif(os.path.islink(i)):
                print(f"XAI image path {i} is a broken or invalid link.")

    def _verify_all_paths(self, config):
        # Catch if the links are missing and raise an exception in case its not found
        not_found = 0
        print(self.CONCEPT_FOLDER_LIST)
        for concept_folder in self.CONCEPT_FOLDER_LIST:
            results = self._verify_files_links(concept_folder)
            if(results == -1):
                print("Folder not found ",concept_folder)
                notfound = 1
        print(self.RANDOM_FOLDER)
        results = self._verify_files_links(self.RANDOM_FOLDER)
        if(results == -1):
            print("Folder not found ",self.RANDOM_FOLDER)
            notfound = 1
        base_path = self.CLASSIFICATION_DATA_BASE_PATH
        for target in self.TARGET_CLASS_LIST:
            folder_path = os.path.join(base_path,"train/"+ target)
            results = self._verify_files_links(folder_path)
        base_path = self.CLASSIFICATION_DATA_BASE_PATH
        for target in self.TARGET_CLASS_LIST:
            folder_path = os.path.join(base_path,"valid/"+ target)
            results = self._verify_files_links(folder_path)
        if(not_found == 1):
            raise(" Some of the folders are not found or an incorrect link ")
        

    def _get_num_classes(self, base_path):
        return len([
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ])
        
        
    def _verify_files_links(self, base_path):
        """
        Checks if the given path is a symbolic link, a regular file, a directory,
        or if it doesn't exist.
        """
        return_value = 0
        if os.path.islink(base_path):
            # The path itself is a symbolic link
            print(f"The path '{base_path}' is a symbolic link.")
            return_value = 0
            # You can also check what the link points to
            if os.path.isdir(base_path):
                print("  It points to a directory.")
                return_value = 0
            elif os.path.isfile(base_path):
                print("  It points to a file.")
                return_value = 0
            else:
                print("  It is a broken or invalid link (target does not exist).")
                return_value = -1
        elif os.path.isdir(base_path):
            print(f"The path '{base_path}' is a regular directory.")
            return_value = 0
        elif os.path.isfile(base_path):
            print(f"The path '{base_path}' is a regular file.")
            return_value = 0
        else:
            print(f"The path '{base_path}' does not exist.")
            return_value = -1
    
        

# Example Usage:
if __name__ == '__main__':
    # Assume 'config.yaml' exists and has the required structure
    # You can create a dummy one for testing:
    # with open('config.yaml', 'w') as f:
    #     yaml.dump({'classification': {'data_base_path': '.', 'target_class_list': [], 'linear_classifier_type': 'linear', 'epochs': 10, 'lambda_aligns': 0.5}, 'concept': {'base_path': '.', 'target_folders': ['concept1', 'concept2'], 'random_folder': ['random1']}}, f)
    parser = argparse.ArgumentParser(description="Run the classification script with a configurable file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (e.g., config.yaml)')
    args = parser.parse_args()
    config_file = args.config

    # First instance creation
    config = ConfigSingleton(config_file)
    print("First instance created.")
    print("Seed:", config.SEED)
    print("Classification data path:", config.CLASSIFICATION_DATA_BASE_PATH)
    print("Target classes:", config.TARGET_CLASS_LIST)
    print("linear classifier type:", config.LINEAR_CLASSIFIER_TYPE)
    print("lambda aligns:", config.LAMBDA_ALIGNS)
    print("concept base path:", config.CONCEPT_FOLDER_LIST)
    print("Random folder path:", config.RANDOM_FOLDER)
    print("concept base path:", config.CONCEPT_FOLDER_LIST)
    print("Learning rate:", config.LEARNING_RATE)
    print("Batch size:", config.BATCH_SIZE)    
    print("Number of classes:", config.NUM_CLASSES)

    print("-" * 20)
    notfound = 0
    for concept_folder in config.CONCEPT_FOLDER_LIST:
        results = config._verify_files_links(concept_folder)
        if(results == -1):
            print("Folder not found ",concept_folder)
            notfound = 1
    print(config.RANDOM_FOLDER)
    
    results = config._verify_files_links(config.RANDOM_FOLDER)
    if(results == -1):
        print("Folder not found ",random_folder)
        notfound = 1
    
    base_path = config.CLASSIFICATION_DATA_BASE_PATH
    for target in config.TARGET_CLASS_LIST:
        folder_path = os.path.join(base_path,"train/"+ target)
        results = config._verify_files_links(folder_path)
        
    base_path = config.CLASSIFICATION_DATA_BASE_PATH
    for target in config.TARGET_CLASS_LIST:
        folder_path = os.path.join(base_path,"valid/"+ target)
        results = config._verify_files_links(folder_path)

    print(f"Recalibration is enabled or not {config.OVERRIDE_RECALIB}")
    print(f"Recalibration Layers of VGG16 {config.VGG_RECALIB}")
    print(f"Recalibration Layers of ResNet50 {config.RESNET50_RECALIB}")
    print(f"Recalibration Layers of Inception V3 {config.INCEPTION_V3_RECALIB}")
    print(f"Recalibration Layers of MobileNet V3 Small {config.MOBILENET_V3_SMALL_RECALIB}")
    print(f"Recalibration Layers of MobileNet V3 Large {config.MOBILENET_V3_LARGE_RECALIB}")
    notfound = 1
    #print(config.XAI_IMAGE_PATH)
    for i in config.XAI_IMAGE_PATH:
        if not (os.path.isfile(i) or os.path.islink(i)):
            print(f"XAI image path {i} does not exist.")
            notfound = notfound + 1
    print(f"Total XAI image paths not found {notfound}")

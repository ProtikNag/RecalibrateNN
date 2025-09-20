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

    def _load_and_process_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
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

    def _get_num_classes(self, base_path):
        return len([
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ])

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
    config1 = ConfigSingleton(config_file)
    print("First instance created.")
    print("Seed:", config1.SEED)
    print("Classification data path:", config1.CLASSIFICATION_DATA_BASE_PATH)
    print("Target classes:", config1.TARGET_CLASS_LIST)
    print("linear classifier type:", config1.LINEAR_CLASSIFIER_TYPE)
    print("lambda aligns:", config1.LAMBDA_ALIGNS)
    print("concept base path:", config1.CONCEPT_FOLDER_LIST)
    print("Random folder path:", config1.RANDOM_FOLDER)
    print("concept base path:", config1.CONCEPT_FOLDER_LIST)
    print("Learning rate:", config1.LEARNING_RATE)
    print("Batch size:", config1.BATCH_SIZE)    
        
    print("Number of classes:", config1.NUM_CLASSES)

    print("-" * 20)

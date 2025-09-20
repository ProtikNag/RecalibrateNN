import argparse
import yaml
import os

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

        self.CLASSIFICATION_DATA_BASE_PATH = config['classification']['data_base_path']
        self.TARGET_CLASS_LIST = config['classification']['target_class_list']
        self.LINEAR_CLASSIFIER_TYPE = config['classification']['linear_classifier_type']
        self.NUM_CLASSES = self._get_num_classes(self.CLASSIFICATION_DATA_BASE_PATH)
        self.EPOCHS = config['classification']['epochs']
        self.LAMBDA_ALIGN = config['classification']['lambda_aligns']

        config_base_path = config['concept']['base_path']
        target_folder_path = config['concept']['target_folders']
        self.CONCEPT_FOLDER_LIST = [os.path.join(config_base_path, concept_name) for concept_name in target_folder_path]

        random_folder_path = config['concept']['random_folder']
        self.RANDOM_FOLDER_LIST = [os.path.join(config_base_path, concept_name) for concept_name in random_folder_path]

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
    print("Number of classes:", config1.NUM_CLASSES)
    print("Concept folders:", config1.CONCEPT_FOLDER_LIST)

    print("-" * 20)

    # Second instance creation - this will return the same object
    config2 = ConfigSingleton(config_file)
    print("Second instance created.")
    print("Are config1 and config2 the same object?", config1 is config2)
    print("Concept folders from second instance:", config2.CONCEPT_FOLDER_LIST)

    # You can access the parameters directly from either instance
    print("Epochs:", config1.EPOCHS)
    print("Epochs:", config2.EPOCHS)
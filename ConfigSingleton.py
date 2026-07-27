# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on```python
# Copyright [2025] [Srikanth KS]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""
/*
 * Copyright (c) 2025 Srikanth K S. All rights reserved.
 * Licensed under the APACHE2 License.
 * Author : Srikanth K S
 * Version 1.0
 */
"""
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
        self._instance._read_multiconcept_variables(self.config)
        self._instance._read_recalibration_weights_variables(self.config)

        
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
        self.XAI_NUMCLASSES = int(config['xai_before_after']['num_classes'])
        self.XAI_IMAGE_PATH = []
        for i in range(0,self.XAI_NUMCLASSES):
            #Read all the xai image paths
            temp  = config['xai_before_after']['class_images']['class'+str(i)]
            valid_images = []
            for j in temp:
                if(os.path.isfile(j)):
                    print(f"XAI image path {j} exists.")
                    valid_images.append(j)
                elif(os.path.islink(j)):
                    print(f"XAI image path {j} is a link.")
                    valid_images.append(j)
            self.XAI_IMAGE_PATH.append(valid_images)

    def _read_multiconcept_variables(self, config):
        """
        Read multiconcept variables from configuration.

        Each class now defines a `random_folders` list positioned 1:1 against
        its `target_folders` list, e.g.:
            class0:
              target_folders:
                - "concept_150/deer/coat"
                - "concept_150/deer/background"
              random_folders:
                - "random"
                - "random"
        The actual random folder used for `target_folders[i]` is
        `random_folder_base_path/random_folders[i]`.
        """
        if 'multiconcept' in config:
            multiconcept_config = config['multiconcept']
            self.MULTICONCEPT_BASE_PATH = multiconcept_config.get('base_path', '')
            self.MULTICONCEPT_RANDOM_FOLDER_BASE_PATH = multiconcept_config.get('random_folder_base_path', '')
            # Legacy schema fallback: a single global random_folder for the whole multiconcept section.
            self.MULTICONCEPT_RANDOM_FOLDER = multiconcept_config.get('random_folder', '')

            # Parse class-based target folders together with their per-folder random folders
            self.MULTICONCEPT_CLASS_CONCEPTS = {}
            self.MULTICONCEPT_CLASS_RANDOM_FOLDERS = {}
            class_idx = 0
            while f'class{class_idx}' in multiconcept_config:
                class_key = f'class{class_idx}'
                class_config = multiconcept_config[class_key]
                target_folders = class_config.get('target_folders', [])
                random_folders = class_config.get('random_folders', [])

                if random_folders and len(random_folders) != len(target_folders):
                    raise ValueError(
                        f"'{class_key}' has {len(target_folders)} target_folders but "
                        f"{len(random_folders)} random_folders; each target folder must "
                        f"have exactly one corresponding random folder (1:1 by position)."
                    )

                self.MULTICONCEPT_CLASS_CONCEPTS[class_idx] = target_folders
                self.MULTICONCEPT_CLASS_RANDOM_FOLDERS[class_idx] = random_folders
                class_idx += 1

            self.MULTICONCEPT_ENABLED = len(self.MULTICONCEPT_CLASS_CONCEPTS) > 0
            self.MULTICONCEPT_NUM_CONCEPTS = class_idx
        else:
            self.MULTICONCEPT_BASE_PATH = ''
            self.MULTICONCEPT_RANDOM_FOLDER_BASE_PATH = ''
            self.MULTICONCEPT_RANDOM_FOLDER = ''
            self.MULTICONCEPT_CLASS_CONCEPTS = {}
            self.MULTICONCEPT_CLASS_RANDOM_FOLDERS = {}
            self.MULTICONCEPT_ENABLED = False
            self.MULTICONCEPT_NUM_CONCEPTS = 0
        return True

    def get_multiconcept_random_folder_path(self, class_idx: int, position_idx: int) -> str:
        """
        Resolve the absolute random-folder path that corresponds to
        MULTICONCEPT_CLASS_CONCEPTS[class_idx][position_idx], using
        random_folder_base_path + the per-position random folder name.
        Falls back to the legacy single MULTICONCEPT_RANDOM_FOLDER if the
        per-position random folder is not defined.
        """
        random_folders = self.MULTICONCEPT_CLASS_RANDOM_FOLDERS.get(class_idx, [])
        if position_idx < len(random_folders):
            return os.path.join(self.MULTICONCEPT_RANDOM_FOLDER_BASE_PATH, random_folders[position_idx])
        return self.MULTICONCEPT_RANDOM_FOLDER

    def _read_recalibration_weights_variables(self, config):
        """
        Read per-class and per-concept recalibration weights used to combine
        multiple concept-alignment losses during targeted recalibration.

        Expected YAML schema (mirrors the multiconcept class0/class1/... style):
            recalibration_weights:
              class0:
                class_weight: 1.0
                concept_weights:
                  deer_coat: 0.4
                  deer_background: 0.2
                  deer_face: 0.2
                  deer_legs: 0.2
              class1:
                class_weight: 1.0
                concept_weights:
                  horse_coat: 0.25
                  ...

        Populates:
            RECALIB_CLASS_WEIGHT       : {class_idx: float}
            RECALIB_CONCEPT_WEIGHTS    : {class_idx: {concept_name: float}}
            RECALIB_WEIGHTS_ENABLED    : bool
        Missing sections/keys fall back to sane defaults (class_weight=1.0,
        equal weight across whatever concepts are actually used at runtime).
        """
        self.RECALIB_CLASS_WEIGHT = {}
        self.RECALIB_CONCEPT_WEIGHTS = {}

        if 'recalibration_weights' in config:
            weights_config = config['recalibration_weights']
            class_idx = 0
            while f'class{class_idx}' in weights_config:
                class_key = f'class{class_idx}'
                class_config = weights_config[class_key] or {}
                self.RECALIB_CLASS_WEIGHT[class_idx] = float(
                    class_config.get('class_weight', 1.0)
                )
                self.RECALIB_CONCEPT_WEIGHTS[class_idx] = {
                    concept_name: float(weight)
                    for concept_name, weight in (class_config.get('concept_weights', {}) or {}).items()
                }
                class_idx += 1
            self.RECALIB_WEIGHTS_ENABLED = class_idx > 0
        else:
            self.RECALIB_WEIGHTS_ENABLED = False
        return True

    def get_class_weight(self, class_idx: int) -> float:
        """Return the configured class-level recalibration weight (default 1.0)."""
        return self.RECALIB_CLASS_WEIGHT.get(class_idx, 1.0)

    def get_concept_weights(self, class_idx: int, concept_names: list) -> dict:
        """
        Return {concept_name: weight} for the given class, restricted to
        `concept_names` actually available at runtime. Any concept missing
        from the config is given an equal share of the remaining weight
        (defaulting to equal weights across all concepts when none are
        configured for this class).
        """
        configured = dict(self.RECALIB_CONCEPT_WEIGHTS.get(class_idx, {}))
        resolved = {}
        unspecified = []
        remaining_weight = 1.0
        for name in concept_names:
            if name in configured:
                resolved[name] = configured[name]
                remaining_weight -= configured[name]
            else:
                unspecified.append(name)
        if unspecified:
            share = max(remaining_weight, 0.0) / len(unspecified)
            for name in unspecified:
                resolved[name] = share
        return resolved

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
        base_path = os.path.join(base_path, "train")
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
    
    # Print multiclass variables if enabled
    if config.MULTICONCEPT_ENABLED:
        print(f"Multiconcept is enabled with {config.MULTICONCEPT_NUM_CONCEPTS} classes")
        print(f"Multiconcept base path: {config.MULTICONCEPT_BASE_PATH}")
        print(f"Multiconcept random folder base path: {config.MULTICONCEPT_RANDOM_FOLDER_BASE_PATH}")
        for class_idx, concepts in config.MULTICONCEPT_CLASS_CONCEPTS.items():
            print(f"  Class {class_idx} concepts: {concepts}")
            for position_idx, concept in enumerate(concepts):
                resolved_random = config.get_multiconcept_random_folder_path(class_idx, position_idx)
                print(f"    {concept}  ->  random_folder: {resolved_random}")
    else:
        print("Multiconcept is not enabled.")
        
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
    notfound = 0
    #print(config.XAI_IMAGE_PATH)
    for classIdx in range(0,config.XAI_NUMCLASSES):
      print(f"Number of XAI images to scan in {classIdx} is {len(config.XAI_IMAGE_PATH[classIdx])}")
      for i in config.XAI_IMAGE_PATH[classIdx]:
          
          if not (os.path.isfile(i) or os.path.islink(i)):
              print(f"XAI image path {i} does not exist.")
              notfound = notfound + 1
    print(f"Total XAI image paths not found {notfound}")

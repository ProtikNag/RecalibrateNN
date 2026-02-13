#---------------------------------------RNet 50 -----------------------------------------------
#BG
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c0_bg --layers_to_pertubate "features.1.block.2.0,features.3.block.2.0,features.4.block.1.0,features.5.block.0.0,features.6.block.3.0,features.7.block.1.0,features.9.block.2.fc2,features.10.block.1.0,features.11.block.3.0,features.12.0"
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c1_bg --layers_to_pertubate "features.1.block.2.0,features.2.block.2.0,features.3.block.1.0,features.4.block.0.0,features.6.block.2.fc2,features.7.block.1.0,features.8.block.2.fc1,features.9.block.0.0,features.10.block.0.0,features.11.block.2.fc2,features.12.0"
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c2_bg --layers_to_pertubate "features.5.block.1.0,features.6.block.2.fc2,features.8.block.0.0,features.9.block.2.fc2,features.11.block.0.0"

#Coat
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c0_coat --layers_to_pertubate "features.1.block.2.0,features.3.block.2.0,features.4.block.1.0,features.5.block.3.0,features.7.block.0.0,features.8.block.3.0,features.9.block.2.fc1,features.10.block.3.0,features.11.block.0.0,features.12.0"
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c1_coat --layers_to_pertubate "features.1.block.2.0,features.2.block.0.0,features.4.block.0.0,features.7.block.0.0,features.8.block.1.0,features.9.block.0.0,features.10.block.1.0,features.11.block.3.0,features.12.0"
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c2_coat --layers_to_pertubate "features.2.block.1.0,features.4.block.1.0,features.5.block.1.0,features.6.block.0.0,features.7.block.0.0,features.8.block.2.fc2,features.9.block.2.fc1,features.10.block.3.0,features.11.block.3.0"

#Face
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c0_face --layers_to_pertubate "features.1.block.0.0,features.2.block.1.0,features.3.block.1.0,features.4.block.1.0,features.5.block.2.fc1,features.6.block.3.0,features.7.block.0.0,features.8.block.3.0,features.9.block.3.0,features.10.block.3.0,features.11.block.3.0,features.12.0" 
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c1_face --layers_to_pertubate "features.1.block.2.0,features.2.block.0.0,features.5.block.3.0,features.7.block.0.0,features.8.block.2.fc1,features.9.block.1.0,features.10.block.3.0,features.11.block.1.0,features.12.0"
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c2_face --layers_to_pertubate "features.2.block.1.0,features.4.block.2.fc2,features.6.block.3.0,features.7.block.2.fc2,features.10.block.3.0,features.11.block.2.fc1,features.12.0"

#Legs
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c0_legs --layers_to_pertubate "features.1.block.2.0,features.2.block.0.0,features.3.block.2.0,features.4.block.1.0,features.5.block.0.0,features.6.block.3.0,features.7.block.0.0,features.9.block.3.0,features.10.block.3.0,features.11.block.3.0,features.12.0"
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c1_legs --layers_to_pertubate "features.1.block.2.0,features.2.block.0.0,features.4.block.0.0,features.5.block.3.0,features.7.block.0.0,features.8.block.2.fc1,features.9.block.3.0,features.10.block.3.0,features.11.block.3.0,features.12.0"
python pertubate_neurons.py --model_name mobilenet_v3_small --config layer_config.yaml --saveas c2_legs --layers_to_pertubate "features.4.block.1.0,features.5.block.1.0,features.6.block.2.fc2,features.7.block.2.fc2,features.10.block.3.0,features.11.block.3.0"


mv moblenet_v3_small*.xlsx /mnt/sdc/sensitivity_analysis_paper/perturbation/mnet_small
mv pert*.log /mnt/sdc/sensitivity_analysis_paper/perturbation/mnet_small

#---------------------------------------RNet 50 -----------------------------------------------

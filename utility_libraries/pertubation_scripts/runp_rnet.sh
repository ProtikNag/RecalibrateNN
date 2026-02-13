#---------------------------------------RNet 50 -----------------------------------------------
#BG
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c0_bg --layers_to_pertubate "layer1.2.conv3,layer2.0.conv3,layer3.0.conv1,layer4.2.conv3"
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c1_bg --layers_to_pertubate "maxpool,layer1.0.conv1,layer2.0.conv2,layer3.5.conv2,layer4.2.conv3"
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c2_bg --layers_to_pertubate "layer3.5.conv2,layer4.2.conv3"

#Coat
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c0_coat --layers_to_pertubate "layer1.2.conv3,layer2.0.conv2,layer3.0.downsample.0,layer4.2.conv3"
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c1_coat --layers_to_pertubate "maxpool,layer1.0.conv1,layer2.0.conv2,layer3.0.downsample.0,layer4.2.conv3"
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c2_coat --layers_to_pertubate "layer1.0.downsample.0,layer2.0.conv3,layer3.5.conv3,layer4.2.conv3"

#Face
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c0_face --layers_to_pertubate "layer1.0.conv3,layer2.0.conv2,layer3.0.conv2,layer4.2.conv3" 
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c1_face --layers_to_pertubate "maxpool,layer1.0.conv1,layer2.0.conv2,layer3.0.downsample.0,layer4.2.conv3"
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c2_face --layers_to_pertubate "layer3.0.conv2,layer4.2.conv2"

#Legs
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c0_legs --layers_to_pertubate "layer1.2.conv3,layer2.3.conv3,layer3.0.conv1"
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c1_legs --layers_to_pertubate "maxpool,layer1.1.conv1,layer2.0.downsample.0,layer3.0.downsample.0,layer4.2.conv3"
python pertubate_neurons.py --model_name resnet50 --config layer_config.yaml --saveas c2_legs --layers_to_pertubate "layer4.0.conv2"


#mv resnet50*.xlsx /mnt/sdc/sensitivity_analysis_paper/perturbation/rnet50
#mv pert*.log /mnt/sdc/sensitivity_analysis_paper/perturbation/rnet50

#---------------------------------------RNet 50 -----------------------------------------------

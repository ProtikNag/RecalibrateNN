#!/bin/bash
# Define layer arrays for each artifact and class combination
declare -A layers=(
    [c0_bg]="layer1.2.conv3,layer2.0.conv3,layer3.0.conv1,layer4.2.conv3"
    [c0_coat]="layer1.2.conv3,layer2.0.conv2,layer3.0.downsample.0,layer4.2.conv3"
    [c0_face]="maxpool,layer1.2.conv3,layer2.0.conv2,layer3.0.conv2,layer4.2.conv3"
    [c0_legs]="layer1.2.conv3,layer2.0.conv2,layer3.0.downsample.0,layer4.0.conv2"

    [c1_bg]="maxpool,layer1.0.conv1,layer2.0.downsample.0,layer3.0.downsample.0,layer4.1.conv3"
    [c1_coat]="maxpool,layer1.0.conv1,layer2.0.downsample.0,layer3.0.downsample.0,layer4.2.conv3"
    [c1_face]="maxpool,layer1.0.conv1,layer2.0.downsample.0,layer3.0.downsample.0,layer4.1.conv3"
    [c1_legs]="maxpool,layer1.0.conv1,layer2.0.downsample.0,layer3.0.downsample.0,layer4.1.conv3"

    [c2_bg]="layer1.2.conv1,layer3.5.conv2,layer4.2.conv3"
    [c2_coat]="layer1.0.downsample.0,layer2.0.conv3,layer3.5.conv3,layer4.1.conv3"
    [c2_face]="layer3.0.conv2,layer4.0.conv2"
    [c2_legs]=" layer4.0.conv3"

)

# Iterate through each configuration
for config in "${!layers[@]}"; do
    python pertubate_neurons.py \
        --model_name resnet50 \
        --config layer_config.yaml \
        --saveas "$config" \
        --layers_to_pertubate "${layers[$config]}"
done


declare -A layers_negative=(
    [c0_bg_negative]="maxpool,layer1.0.conv1,layer2.1.conv2,layer3.1.conv1,layer4.2.conv1"
    [c0_coat_negative]="maxpool,layer1.1.conv2,layer2.1.conv3,layer3.1.conv1,layer4.2.conv1"
    [c0_face_negative]="conv1,layer1.2.conv1,layer2.1.conv3,layer3.1.conv1,layer4.1.conv2"
    [c0_legs_negative]="maxpool,layer1.1.conv1,layer2.1.conv3,layer3.0.conv3,layer4.0.downsample.0"

    [c1_bg_negative]="conv1,layer1.2.conv1,layer2.3.conv3,layer3.2.conv1,layer4.1.conv2"
    [c1_coat_negative]="conv1,layer1.2.conv1,layer2.3.conv3,layer3.0.conv1,layer4.1.conv2"
    [c1_face_negative]="conv1,layer1.2.conv1,layer2.1.conv1,layer3.2.conv1"
    [c1_legs_negative]="conv1,layer1.2.conv1,layer2.0.conv1,layer3.2.conv1"

    [c2_bg_negative]="maxpool,layer1.0.conv1,layer2.0.conv1,layer3.0.downsample.0,layer4.0.downsample.0"
    [c2_coat_negative]="maxpool,layer1.0.conv1,layer2.1.conv2,layer3.0.downsample.0"
    [c2_face_negative]="maxpool,layer1.0.conv1,layer2.0.conv2,layer3.0.conv1,layer4.2.conv1"
    [c2_legs_negative]="maxpool,layer1.0.conv1,layer2.0.conv1,layer3.0.conv1,layer4.1.conv2"

)

# Iterate through each configuration
for config in "${!layers_negative[@]}"; do
    python pertubate_neurons.py \
        --model_name resnet50 \
        --config layer_config.yaml \
        --saveas "$config" \
        --layers_to_pertubate "${layers_negative[$config]}"
done


mv resnet50*.xlsx /mnt/sdc/sensitivity_analysis_paper/perturbation/rnet50
mv pert*.log /mnt/sdc/sensitivity_analysis_paper/perturbation/rnet50

#---------------------------------------RNet 50 -----------------------------------------------

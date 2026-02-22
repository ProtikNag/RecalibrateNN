#!/bin/bash
MODEL="mobilenet_v3_large"
# Define layer arrays for each artifact and class combination
##Inverted Residue .10, Inverted Residue .11 Inverted Residue .12 Inverted Residue .4 
##Inverted Residue .5 Inverted Residue .6 Inverted Residue .7 Inverted Residue .8 Inverted Residue .9 
##Inverted Residue .1 Inverted Residue .2 Inverted Residue .3 Stem
declare -A layers=(
    [c0_bg]="features.10.block.1.0,features.11.block.0.0,features.12.block.3.0,features.13.block.0.0,features.14.block.3.0,features.15.block.3.0,features.16.0,features.2.block.2.0,features.3.block.1.0,features.4.block.3.0,features.5.block.2.fc2,features.7.block.1.0,features.9.block.2.0"
    [c0_coat]="features.11.block.0.0,features.12.block.3.0,features.13.block.1.0,features.14.block.3.0,features.15.block.3.0,features.16.0,features.2.block.2.0,features.4.block.3.0,features.5.block.2.fc1,features.7.block.1.0,features.8.block.0.0"
    [c0_face]="features.10.block.2.0,features.11.block.3.0,features.12.block.1.0,features.13.block.3.0,features.14.block.3.0,features.15.block.3.0,features.16.0,features.2.block.1.0,features.4.block.3.0,features.7.block.1.0,features.8.block.2.0,features.9.block.2.0"
    [c0_legs]="features.11.block.0.0,features.12.block.3.0,features.13.block.1.0,features.14.block.3.0,features.15.block.3.0,features.16.0,features.2.block.2.0,features.3.block.1.0,features.4.block.3.0,features.7.block.0.0,features.8.block.2.0,features.9.block.0.0"

    [c1_bg]="features.1.block.1.0,features.10.block.2.0,features.11.block.2.fc2,features.12.block.2.fc2,features.13.block.2.fc1,features.15.block.2.fc2,features.2.block.0.0,features.3.block.1.0,features.4.block.1.0,features.5.block.1.0,features.6.block.3.0"
    [c1_coat]="features.1.block.1.0,features.10.block.2.0,features.11.block.2.fc2,features.12.block.2.fc1,features.13.block.2.fc1,features.15.block.2.fc1,features.2.block.2.0,features.3.block.0.0,features.5.block.1.0,features.6.block.2.fc1"
    [c1_face]="features.1.block.1.0,features.10.block.1.0,features.11.block.2.fc1,features.12.block.2.fc2,features.13.block.2.fc1,features.15.block.3.0,features.16.0,features.2.block.2.0,features.5.block.0.0"
    [c1_legs]="features.1.block.1.0,features.10.block.2.0,features.12.block.3.0,features.13.block.2.fc1,features.15.block.2.fc1,features.2.block.1.0,features.4.block.1.0,features.5.block.0.0,features.6.block.2.fc1"

    [c2_bg]="features.1.block.0.0,features.12.block.1.0,features.13.block.3.0,features.14.block.2.fc2,features.15.block.3.0,features.3.block.0.0,features.5.block.3.0,features.6.block.3.0,features.8.block.1.0"
    [c2_coat]="features.1.block.0.0,features.11.block.0.0,features.12.block.2.fc2,features.13.block.3.0,features.14.block.3.0,features.15.block.3.0,features.16.0,features.2.block.2.0,features.3.block.1.0,features.4.block.2.fc2,features.5.block.1.0,features.6.block.2.fc2,features.7.block.2.0,features.8.block.1.0"
    [c2_face]="features.1.block.0.0,features.10.block.1.0,features.11.block.0.0,features.13.block.3.0,features.14.block.3.0,features.15.block.3.0,features.16.0,features.3.block.0.0,features.4.block.2.fc2,features.5.block.2.fc2,features.6.block.2.fc2,features.8.block.2.0"
    [c2_legs]="features.10.block.0.0,features.13.block.3.0,features.14.block.3.0,features.15.block.3.0,features.16.0,features.3.block.0.0,features.4.block.2.fc2,features.5.block.2.fc2,features.6.block.2.fc2,features.8.block.1.0,features.9.block.2.0"

)

# Iterate through each configuration
for config in "${!layers[@]}"; do
    python pertubate_neurons.py \
        --model_name "$MODEL" \
        --config layer_config.yaml \
        --saveas "$config" \
        --layers_to_pertubate "${layers[$config]}"
done


declare -A layers_negative=(
    [c0_bg_negative]="features.1.block.0.0,features.10.block.2.0,features.11.block.2.fc1,features.12.block.2.fc2,features.13.block.2.fc1,features.15.block.2.fc2,features.2.block.0.0,features.3.block.0.0,features.4.block.2.fc2,features.5.block.3.0,features.6.block.0.0,features.8.block.1.0,features.9.block.0.0,features.0.0"
    [c0_coat_negative]="features.1.block.1.0,features.10.block.1.0,features.11.block.2.fc2,features.12.block.2.fc2,features.13.block.2.fc1,features.15.block.2.fc1,features.2.block.0.0,features.3.block.2.0,features.4.block.2.fc2,features.5.block.3.0,features.6.block.0.0,features.8.block.1.0,features.9.block.0.0,features.0.0"
    [c0_face_negative]="features.1.block.1.0,features.10.block.0.0,features.11.block.2.fc2,features.12.block.2.fc2,features.13.block.2.fc1,features.14.block.1.0,features.15.block.2.fc2,features.2.block.2.0,features.3.block.0.0,features.4.block.2.fc2,features.5.block.2.fc2,features.6.block.2.fc1,features.8.block.1.0,features.9.block.0.0,features.0.0"
    [c0_legs_negative]="features.1.block.1.0,features.10.block.0.0,features.11.block.2.fc2,features.12.block.2.fc2,features.13.block.2.fc2,features.15.block.2.fc2,features.3.block.0.0,features.4.block.2.fc2,features.5.block.3.0,features.6.block.0.0,features.8.block.1.0,features.9.block.2.0,features.0.0"

    [c1_bg_negative]="features.1.block.0.0,features.11.block.3.0,features.12.block.3.0,features.13.block.0.0,features.14.block.1.0,features.15.block.0.0,features.16.0,features.2.block.2.0,features.3.block.2.0,features.4.block.3.0,features.5.block.2.fc1,features.6.block.2.fc2,features.7.block.0.0,features.8.block.2.0,features.9.block.1.0,features.0.0"
    [c1_coat_negative]="features.1.block.0.0,features.11.block.3.0,features.12.block.1.0,features.13.block.0.0,features.14.block.0.0,features.15.block.1.0,features.16.0,features.2.block.0.0,features.3.block.2.0,features.4.block.3.0,features.5.block.2.fc1,features.6.block.3.0,features.7.block.0.0,features.8.block.2.0,features.9.block.1.0,features.0.0"
    [c1_face_negative]="features.1.block.0.0,features.11.block.2.fc2,features.12.block.0.0,features.13.block.2.fc2,features.14.block.0.0,features.15.block.1.0,features.3.block.2.0,features.4.block.3.0,features.5.block.2.fc2,features.6.block.2.fc2,features.7.block.2.0,features.8.block.1.0,features.9.block.0.0,features.0.0"
    [c1_legs_negative]="features.1.block.0.0,features.11.block.3.0,features.12.block.1.0,features.13.block.1.0,features.14.block.0.0,features.15.block.1.0,features.16.0,features.3.block.2.0,features.4.block.0.0,features.5.block.2.fc2,features.6.block.0.0,features.7.block.0.0,features.8.block.1.0,features.9.block.2.0,features.0.0"

    [c2_bg_negative]="features.1.block.1.0,features.10.block.1.0,features.11.block.0.0,features.12.block.2.fc1,features.13.block.0.0,features.14.block.3.0,features.15.block.1.0,features.16.0,features.2.block.1.0,features.4.block.1.0,features.5.block.2.fc1,features.6.block.0.0,features.7.block.0.0,features.8.block.2.0,features.9.block.2.0,features.0.0"
    [c2_coat_negative]="features.1.block.1.0,features.10.block.1.0,features.11.block.2.fc1,features.12.block.3.0,features.13.block.2.fc2,features.14.block.2.fc1,features.15.block.2.fc1,features.2.block.1.0,features.4.block.2.fc1,features.5.block.3.0,features.6.block.1.0,features.8.block.2.0,features.9.block.1.0,features.0.0"
    [c2_face_negative]="features.1.block.1.0,features.11.block.2.fc1,features.12.block.0.0,features.13.block.2.fc2,features.14.block.1.0,features.15.block.2.fc1,features.2.block.0.0,features.3.block.2.0,features.4.block.0.0,features.5.block.0.0,features.6.block.0.0,features.7.block.0.0,features.8.block.0.0,features.9.block.1.0,features.0.0"
    [c2_legs_negative]="features.1.block.1.0,features.10.block.2.0,features.11.block.0.0,features.12.block.1.0,features.13.block.2.fc1,features.14.block.1.0,features.15.block.0.0,features.2.block.0.0,features.3.block.1.0,features.4.block.2.fc1,features.5.block.2.fc1,features.6.block.3.0,features.7.block.2.0,features.8.block.2.0,features.9.block.1.0,features.0.0"
)

# Iterate through each configuration
for config in "${!layers_negative[@]}"; do
    python pertubate_neurons.py \
        --model_name "$MODEL" \
        --config layer_config.yaml \
        --saveas "$config" \
        --layers_to_pertubate "${layers_negative[$config]}"
done

mv ${MODEL}*.xlsx /mnt/sdc/sensitivity_analysis_paper/perturbation/${MODEL}
mv pert*.log /mnt/sdc/sensitivity_analysis_paper/perturbation/${MODEL}


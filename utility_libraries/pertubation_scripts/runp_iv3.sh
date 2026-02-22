#!/bin/bash
MODEL="inception_v3"
# Define layer arrays for each artifact and class combination
##Inception A Inception B Inception C Reduction A Reduction B Stem
declare -A layers=(
    [c0_bg]="Mixed_5d.branch3x3dbl_3.conv,Mixed_6e.branch_pool.conv,Mixed_7c.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_7a.branch7x7x3_2.conv,Conv2d_3b_1x1.conv"
    [c0_coat]="Mixed_5b.branch1x1.conv,Mixed_6e.branch_pool.conv,Mixed_7c.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_7a.branch7x7x3_2.conv,Conv2d_3b_1x1.conv"
    [c0_face]="Mixed_5c.branch1x1.conv,Mixed_6e.branch1x1.conv,Mixed_7b.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_7a.branch3x3_2.conv,Conv2d_3b_1x1.conv"
    [c0_legs]="Mixed_5d.branch1x1.conv,Mixed_6e.branch1x1.conv,Mixed_7c.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_7a.branch3x3_2.conv,maxpool1"

    [c1_bg]="Mixed_5c.branch5x5_1.conv,Mixed_6e.branch7x7dbl_3.conv,Mixed_7c.branch1x1.conv,Mixed_7a.branch7x7x3_2.conv,Conv2d_4a_3x3.conv"
    [c1_coat]="Mixed_5c.branch5x5_1.conv,Mixed_6e.branch7x7dbl_5.conv,Mixed_7c.branch1x1.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_7a.branch7x7x3_4.conv,Conv2d_3b_1x1.conv"
    [c1_face]="Mixed_5c.branch5x5_2.conv,Mixed_6e.branch7x7dbl_5.conv,Mixed_7b.branch1x1.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_7a.branch7x7x3_4.conv,Conv2d_4a_3x3.conv"
    [c1_legs]="Mixed_5c.branch5x5_1.conv,Mixed_6e.branch7x7dbl_5.conv,Mixed_7b.branch3x3dbl_3b.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_7a.branch7x7x3_4.conv,Conv2d_4a_3x3.conv"

    [c2_bg]="Mixed_5c.branch_pool.conv,Mixed_6d.branch7x7dbl_5.conv,Mixed_7c.branch3x3_2a.conv,Mixed_6a.branch3x3dbl_1.conv,Mixed_7a.branch7x7x3_2.conv,Conv2d_2a_3x3.conv"
    [c2_coat]="Mixed_5d.branch3x3dbl_3.conv,Mixed_6e.branch_pool.conv,Mixed_7c.branch1x1.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_7a.branch7x7x3_4.conv,maxpool2"
    [c2_face]="Mixed_5c.branch1x1.conv,Mixed_6e.branch1x1.conv,Mixed_7b.branch1x1.conv,Mixed_6a.branch3x3.conv,maxpool2"
    [c2_legs]="Mixed_5c.branch_pool.conv,Mixed_6e.branch1x1.conv,Mixed_7b.branch3x3_1.conv,Mixed_6a.branch3x3dbl_3.conv,Conv2d_2b_3x3.conv"

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
    [c0_bg_negative]="Mixed_5b.branch_pool.conv,Mixed_6b.branch7x7dbl_5.conv,Mixed_7b.branch1x1.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_7a.branch3x3_2.conv,Conv2d_4a_3x3.conv"
    [c0_coat_negative]="Mixed_5b.branch_pool.conv,Mixed_6b.branch7x7dbl_1.conv,Mixed_7b.branch3x3dbl_1.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_7a.branch3x3_1.conv,Conv2d_2a_3x3.conv"
    [c0_face_negative]="Mixed_5b.branch_pool.conv,Mixed_6d.branch7x7dbl_1.conv,Mixed_7b.branch3x3dbl_1.conv,Mixed_6a.branch3x3dbl_1.conv,Mixed_7a.branch7x7x3_1.conv,Conv2d_1a_3x3.conv"
    [c0_legs_negative]="Mixed_5b.branch_pool.conv,Mixed_6d.branch7x7_3.conv,Mixed_7b.branch3x3dbl_1.conv,Mixed_6a.branch3x3dbl_3.conv,Conv2d_1a_3x3.conv"

    [c1_bg_negative]="Mixed_5b.branch1x1.conv,Mixed_6b.branch1x1.conv,Mixed_7b.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_7a.branch7x7x3_1.conv,Conv2d_2b_3x3.conv"
    [c1_coat_negative]="Mixed_5c.branch1x1.conv,Mixed_6b.branch1x1.conv,Mixed_7b.branch3x3dbl_2.conv,Mixed_6a.branch3x3.conv,Mixed_7a.branch3x3_2.conv,Conv2d_2b_3x3.conv"
    [c1_face_negative]="Mixed_5c.branch1x1.conv,Mixed_6b.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_7a.branch7x7x3_1.conv,Conv2d_2b_3x3.conv"
    [c1_legs_negative]="Mixed_5c.branch1x1.conv,Mixed_6b.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_7a.branch7x7x3_1.conv,Conv2d_2b_3x3.conv"

    [c2_bg_negative]="Mixed_5c.branch1x1.conv,Mixed_6b.branch1x1.conv,Mixed_7b.branch1x1.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_7a.branch3x3_1.conv,maxpool1"
    [c2_coat_negative]="Mixed_5b.branch_pool.conv,Mixed_6e.branch7x7_1.conv,Mixed_7b.branch3x3dbl_1.conv,Mixed_6a.branch3x3dbl_2.conv,Mixed_7a.branch3x3_1.conv,Conv2d_3b_1x1.conv"
    [c2_face_negative]="Mixed_5c.branch5x5_2.conv,Mixed_6c.branch7x7dbl_2.conv,Mixed_7b.branch3x3dbl_1.conv,Mixed_6a.branch3x3dbl_2.conv,Mixed_7a.branch3x3_1.conv,maxpool1"
    [c2_legs_negative]="Mixed_5c.branch3x3dbl_3.conv,Mixed_6c.branch7x7_1.conv,Mixed_7b.branch3x3dbl_1.conv,Mixed_6a.branch3x3dbl_2.conv,Mixed_7a.branch3x3_1.conv,maxpool1"
)

# Iterate through each configuration
for config in "${!layers_negative[@]}"; do
    python pertubate_neurons.py \
        --model_name "$MODEL" \
        --config layer_config.yaml \
        --saveas "$config" \
        --layers_to_pertubate "${layers_negative[$config]}"
done

#!/bin/bash

for model in vgg16; do
    for animal in deer horse zebra; do
        case $model in
            resnet50)
                layer="layer4.0.conv2"
                ;;
            mobilenet_v3_small)
                layer="features.9.block.1.0"
                ;;
            mobilenet_v3_large)
                layer="features.13.block.2.fc1"
                ;;
            inception_v3)
                layer="Mixed_6e.branch1x1.conv"
                ;;
            vgg16)
                layer="features.17"
                ;;
        esac
        
        python simple_sample_cav.py --model-path "/mnt/sdd/basics/balanced_training/$model/$model.pth" --layer-name "$layer" --concept-folder "/mnt/sdc/concepts/concepts_links/concept_150/$animal/coat" --random-folder "/mnt/sdc/concepts/concepts_links/concept_150/random" --base-model "$model" --sample-size 30 --batch-size 10 --seed 141
        mkdir -p /home/srikanth/study1/RecalibrateNN/samplesize_exp2 
        mv concept_data_check_${model}.csv /home/srikanth/study1/RecalibrateNN/samplesize_exp2/concept_data_check_${model}_${animal}coat.csv
    done
done

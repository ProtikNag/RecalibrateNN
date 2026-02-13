#---------------------------------------IV3 -----------------------------------------------
#BG
python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c0_bg --layers_to_pertubate "maxpool2,Mixed_5d.branch3x3dbl_3.conv,Mixed_6e.branch_pool.conv,Mixed_7a.branch7x7x3_2.conv,Mixed_7c.branch1x1.conv"

python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c1_bg --layers_to_pertubate "Conv2d_4a_3x3.conv,Mixed_5c.branch5x5_1.conv,Mixed_6e.branch7x7dbl_3.conv,Mixed_7a.branch7x7x3_2.conv,Mixed_7c.branch3x3dbl_3a.conv"

python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c2_bg --layers_to_pertubate "Conv2d_2a_3x3.conv,Mixed_5c.branch_pool.conv,Mixed_6a.branch3x3dbl_1.conv,Mixed_6d.branch7x7dbl_5.conv,Mixed_7a.branch7x7x3_2.conv,Mixed_7c.branch3x3_2a.conv"

#Coat
python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c0_coat --layers_to_pertubate "Conv2d_3b_1x1.conv,Mixed_5b.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_6e.branch_pool.conv,Mixed_7a.branch7x7x3_2.conv,Mixed_7c.branch3x3_1.conv"

python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c1_coat --layers_to_pertubate "Conv2d_3b_1x1.conv,Mixed_5c.branch5x5_1.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_6e.branch7x7dbl_5.conv,Mixed_7a.branch7x7x3_4.conv,Mixed_7c.branch3x3_2a.conv"

python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c2_coat --layers_to_pertubate "maxpool2,Mixed_5d.branch3x3dbl_3.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_6e.branch1x1.conv,Mixed_7a.branch7x7x3_4.conv,Mixed_7c.branch3x3_2b.conv"

#Face
python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c0_face --layers_to_pertubate "Conv2d_3b_1x1.conv,Mixed_5c.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_6e.branch1x1.conv,Mixed_7a.branch3x3_2.conv,Mixed_7c.branch1x1.conv"

python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c1_face --layers_to_pertubate "Conv2d_4a_3x3.conv,Mixed_5c.branch5x5_2.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_6e.branch7x7dbl_5.conv,Mixed_7a.branch7x7x3_4.conv,Mixed_7c.branch1x1.conv"

python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c2_face --layers_to_pertubate "maxpool2,Mixed_5c.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_6d.branch1x1.conv,Mixed_7b.branch3x3_2b.conv"

#Legs
python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c0_legs --layers_to_pertubate "maxpool1,Mixed_5d.branch1x1.conv,Mixed_6a.branch3x3.conv,Mixed_6d.branch1x1.conv,Mixed_7a.branch3x3_2.conv,Mixed_7c.branch3x3dbl_3a.conv"

python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c1_legs --layers_to_pertubate "Conv2d_4a_3x3.conv,Mixed_5c.branch5x5_1.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_6e.branch7x7dbl_5.conv,Mixed_7a.branch3x3_2.conv,Mixed_7c.branch3x3dbl_3b.conv"

python pertubate_neurons.py --model_name inception_v3 --config layer_config.yaml --saveas c2_legs --layers_to_pertubate "Conv2d_2b_3x3.conv,Mixed_5c.branch_pool.conv,Mixed_6a.branch3x3dbl_3.conv,Mixed_6e.branch1x1.conv,Mixed_7b.branch3x3_2b.conv"


mkdir -p /mnt/sdc/sensitivity_analysis_paper/perturbation/iv3_new
mv *.xlsx /mnt/sdc/sensitivity_analysis_paper/perturbation/iv3_new
mv pert*.log /mnt/sdc/sensitivity_analysis_paper/perturbation/iv3_new

#---------------------------------------RNet 50 -----------------------------------------------

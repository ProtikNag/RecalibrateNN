python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath ./results/ --config config_cub_3classes.yaml

#python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath /mnt/sdd/caltech/results/ --config config_cub_3classes.yaml

python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/resnet50/resnet50.pth"  --model_name "resnet50" --recal_model_basepath /mnt/sdd/caltech/results/ --config config_cub_3classes.yaml


python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/inceptionv3/inceptionv3.pth"  --model_name "inceptionv3" --recal_model_basepath /mnt/sdd/caltech/results/ --config config_cub_3classes.yaml

python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/mobilenet_v3_small/mobilenet_v3_small.pth"  --model_name "mobilenet_v3_small" --recal_model_basepath /mnt/sdd/caltech/results/ --config config_cub_3classes.yaml

python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/mobilenet_v3_large/mobilenet_v3_large.pth"  --model_name "mobilenet_v3_large" --recal_model_basepath /mnt/sdd/caltech/results/ --config config_cub_3classes.yaml



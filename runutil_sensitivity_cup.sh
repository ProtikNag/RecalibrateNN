python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath ./results_3classes/ --config config_cub_3classes.yaml --store_results ./results_3classes/vgg16

#python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath /mnt/sdd/caltech/results/ --config config_cub_3classes.yaml

python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/resnet50/resnet50.pth"  --model_name "resnet50" --recal_model_basepath ./results_3classes/ --config config_cub_3classes.yaml  --store_results ./results_3classes/resnet50


python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/inception_v3/inception_v3.pth"  --model_name "inception_v3" --recal_model_basepath ./results_3classes/ --config config_cub_3classes.yaml --store_results ./results_3classes/inception_v3

python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/mobilenet_v3_small/mobilenet_v3_small.pth"  --model_name "mobilenet_v3_small" --recal_model_basepath ./results_3classes/ --config config_cub_3classes.yaml  --store_results ./results_3classes/mobilenet_v3_small

python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/mobilenet_v3_large/mobilenet_v3_large.pth"  --model_name "mobilenet_v3_large" --recal_model_basepath ./results_3classes/ --config config_cub_3classes.yaml  --store_results ./results_3classes/mobilenet_v3_large



export PLATFORM="CUB"
#python util_sensitivity_compute.py --org_model_path "/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth" --modified_model_path "/mnt/data/results/vgg16/loss_vgg16_features.17_0.3.pth" --model_name "vgg16" --layer_name "features.17"

python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_multiclass/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath /mnt/sdd/caltech/multiclass/results/
python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_multiclass/resnet50/resnet50.pth"  --model_name "resnet50" --recal_model_basepath /mnt/sdd/caltech/multiclass/results/
python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_multiclass/inceptionv3/inceptionv3.pth"  --model_name "inceptionv3" --recal_model_basepath /mnt/sdd/caltech/multiclass/results/
python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_multiclass/mobilenet_v3_small/mobilenet_v3_small.pth"  --model_name "mobilenet_v3_small" --recal_model_basepath /mnt/sdd/caltech/multiclass/results/
python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_multiclass/mobilenet_v3_large/mobilenet_v3_large.pth"  --model_name "mobilenet_v3_large" --recal_model_basepath /mnt/sdd/caltech/multiclass/results/


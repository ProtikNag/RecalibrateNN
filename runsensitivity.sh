export PLATFORM="Srikanth"
python util_sensitivity_compute.py --model_name vgg16   --org_model_path /home/srikanth/trained_models/pytorch/vgg16/vgg16.pth --recal_model_basepath /mnt/data/results/
python util_sensitivity_compute.py --model_name resnet50 --org_model_path /home/srikanth/trained_models/pytorch/resnet50/resnet50.pth --recal_model_basepath /mnt/data/results/
python util_sensitivity_compute.py --model_name inception_v3 --org_model_path /home/srikanth/trained_models/pytorch/inception_v3/inception_v3.pth --recal_model_basepath /mnt/data/results/
#python util_sensitivity_compute.py --model_name mobilenet_v3_small --org_model_path /home/srikanth/trained_models/pytorch/mobilenet_v3_small/mobilenet_v3_small.pth --recal_model_basepath /mnt/data/results/
#python util_sensitivity_compute.py --model_name mobilenet_v3_large --org_model_path /home/srikanth/trained_models/pytorch/mobilenet_v3_large/mobilenet_v3_large.pth --recal_model_basepath /mnt/data/results/

#python util_sensitivity_compute.py --model_name mobilenet_v3_large --model_path /home/srikanth/trained_models/pytorch/mobilenet_v3_large/mobilenet_v3_large.pth --recal_model_basepath /mnt/data/results/

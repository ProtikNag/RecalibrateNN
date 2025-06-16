export PLATFORM="Srikanth"
python main_compute_sensitivity.py --model_name vgg16  --model_path /home/srikanth/trained_models/pytorch
python main_compute_sensitivity.py --model_name resnet50 --model_path /home/srikanth/trained_models/pytorch
python main_compute_sensitivity.py --model_name inception_v3 --model_path /home/srikanth/trained_models/pytorch
python main_compute_sensitivity.py --model_name mobilenet_v3_small --model_path /home/srikanth/trained_models/pytorch
python main_compute_sensitivity.py --model_name mobilenet_v3_large --model_path /home/srikanth/trained_models/pytorch

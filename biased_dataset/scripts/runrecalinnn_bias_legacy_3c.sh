export PATH=$PATH:/home/srikanth/study1/RecalibrateNN/biased_dataset/config
python ../../main.py --model_name vgg16  --model_path /home/srikanth/biased_dataset --config ../config/legacy/config_biased_legacy_3classes.yaml  --store_results /home/srikanth/biased_dataset/recalib/
python ../../main.py --model_name resnet50 --model_path /home/srikanth/biased_dataset --config ../config/legacy/config_biased_legacy_3classes.yaml  --store_results /home/srikanth/biased_dataset/recalib
python ../../main.py --model_name inception_v3 --model_path /home/srikanth/biased_dataset --config ../config/legacy/config_biased_legacy_3classes.yaml  --store_results /home/srikanth/biased_dataset/recalib
python ../../main.py --model_name mobilenet_v3_small --model_path /home/srikanth/biased_dataset --config ../config/legacy/config_biased_legacy_3classes.yaml  --store_results /home/srikanth/biased_dataset/recalib
python ../../main.py --model_name mobilenet_v3_large --model_path /home/srikanth/biased_dataset --config ../config/legacy/config_biased_legacy_3classes.yaml  --store_results /home/srikanth/biased_dataset/recalib

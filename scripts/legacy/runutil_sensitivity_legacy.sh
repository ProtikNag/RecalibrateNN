export CONFIG="../../config_legacy_3classes.yaml"
export PYTHON_SCRIPT="../../util_sensitivity_compute.py"


#python util_sensitivity_compute.py --org_model_path "/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth" --modified_model_path "/mnt/data/results/vgg16/loss_vgg16_features.17_0.3.pth" --model_name "vgg16" --layer_name "features.17"

python ${PYTHON_SCRIPT}  --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/legacy/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath /mnt/sdd/basics/ --store_results   /mnt/sdd/basics/ --config ${CONFIG}

python ${PYTHON_SCRIPT} --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/legacy/resnet50/resnet50.pth"  --model_name "resnet50" --recal_model_basepath /mnt/sdd/basics/ --store_results  /mnt/sdd/basics/ --config ${CONFIG}


python ${PYTHON_SCRIPT}  --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/legacy/inception_v3/inception_v3.pth"  --model_name "inception_v3" --recal_model_basepath /mnt/sdd/basics/ --store_results  /mnt/sdd/basics/ --config ${CONFIG}

python ${PYTHON_SCRIPT}  --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/legacy/mobilenet_v3_small/mobilenet_v3_small.pth"  --model_name "mobilenet_v3_small" --recal_model_basepath /mnt/sdd/basics/ --store_results  /mnt/sdd/basics/ --config ${CONFIG}

python ${PYTHON_SCRIPT}  --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/legacy/mobilenet_v3_large/mobilenet_v3_large.pth"  --model_name "mobilenet_v3_large" --recal_model_basepath /mnt/sdd/basics/ --store_results  /mnt/sdd/basics/ --config ${CONFIG}



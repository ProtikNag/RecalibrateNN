

export CONFIG="../../config_cub_multiclass.yaml"
export PYTHON_SCRIPT="../../util_sensitivity_compute.py"
python  ${PYTHON_SCRIPT} --before_after  --org_model_path "/mnt/data/trained_model/pytorch/caltech_10c_models/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath /mnt/sdd/cub_10c/  --store_results  /mnt/sdd/cub_10c/  --config ${CONFIG}


python  ${PYTHON_SCRIPT} --before_after  --org_model_path "/mnt/data/trained_model/pytorch/caltech_10c_models/resnet50/resnet50.pth"  --model_name "resnet50" --recal_model_basepath /mnt/sdd/cub_10c/ --store_results  /mnt/sdd/cub_10c/  --config ${CONFIG}

python   ${PYTHON_SCRIPT} --before_after  --org_model_path "/mnt/data/trained_model/pytorch/caltech_10c_models/inception_v3/inception_v3.pth"  --model_name "inception_v3" --recal_model_basepath /mnt/sdd/cub_10c/ --store_results  /mnt/sdd/cub_10c/  --config ${CONFIG}


python   ${PYTHON_SCRIPT}  --before_after  --org_model_path "/mnt/data/trained_model/pytorch/caltech_10c_models/mobilenet_v3_small/mobilenet_v3_small.pth"  --model_name "mobilenet_v3_small" --recal_model_basepath /mnt/sdd/cub_10c/ --store_results  /mnt/sdd/cub_10c/  --config ${CONFIG}


python   ${PYTHON_SCRIPT}  --before_after  --org_model_path "/mnt/data/trained_model/pytorch/caltech_10c_models/mobilenet_v3_large/mobilenet_v3_large.pth"  --model_name "mobilenet_v3_large" --recal_model_basepath /mnt/sdd/cub_10c/ --store_results  /mnt/sdd/cub_10c/  --config ${CONFIG}


python   ${PYTHON_SCRIPT}  --org_model_path "/mnt/data/trained_model/pytorch/caltech_10c_models/inception_v3/inception_v3.pth"  --model_name "inception_v3" --recal_model_basepath /mnt/sdd/cub_10c/ --store_results  /mnt/sdd/cub_10c/  --config ${CONFIG}

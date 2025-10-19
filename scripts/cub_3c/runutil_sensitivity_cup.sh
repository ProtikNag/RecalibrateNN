export CONFIG="../../config_cub_3classes.yaml"
export PYTHON_SCRIPT="../../util_sensitivity_compute.py"

python ${PYTHON_SCRIPT} --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_3class/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath /mnt/sdd/cub_3c/ --config ${CONFIG} --store_results  /mnt/sdd/cub_3c/results_3classes/vgg16

#python util_sensitivity_compute.py --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pth"  --model_name "vgg16" --recal_model_basepath /mnt/sdd/caltech/results/ --config config_cub_3classes.yaml

python ${PYTHON_SCRIPT} --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_3class/resnet50/resnet50.pth"  --model_name "resnet50" --recal_model_basepath /mnt/sdd/cub_3c/ --config ${CONFIG}  --store_results  /mnt/sdd/cub_3c/resnet50


python ${PYTHON_SCRIPT} --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_3class/inception_v3/inception_v3.pth"  --model_name "inception_v3" --recal_model_basepath /mnt/sdd/cub_3c/ --config ${CONFIG} --store_results  /mnt/sdd/cub_3c/inception_v3

python ${PYTHON_SCRIPT} --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_3class/mobilenet_v3_small/mobilenet_v3_small.pth"  --model_name "mobilenet_v3_small" --recal_model_basepath /mnt/sdd/cub_3c/ --config ${CONFIG}  --store_results  /mnt/sdd/cub_3c/mobilenet_v3_small

python ${PYTHON_SCRIPT} --before_after  --org_model_path "/home/srikanth/trained_models/pytorch/caltech_3class/mobilenet_v3_large/mobilenet_v3_large.pth"  --model_name "mobilenet_v3_large" --recal_model_basepath /mnt/sdd/cub_3c/ --config ${CONFIG}  --store_results /mnt/sdd/cub_3c/mobilenet_v3_large



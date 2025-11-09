
#########################################################

mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_all/vgg16
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_all/resnet50
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_all/inception_v3
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_all/mobilenet_v3_small
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_all/mobilenet_v3_large

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/vgg16/sensitivity_audit_trail_vgg16_20251027_074803.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/vgg16/corelation_all_vgg16.xlsx


python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/resnet50/sensitivity_audit_trail_resnet50_20251027_081738.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/resnet50/corelation_all_resnet50.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/inception_v3/sensitivity_audit_trail_inception_v3_20251027_091150.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/inception_v3/corelation_all_inception_v3.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/mobilenet_v3_large/sensitivity_audit_trail_mobilenet_v3_large_20251027_110826.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/mobilenet_v3_large/corelation_all_mobilenet_v3_large.xlsx


python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/mobilenet_v3_small/sensitivity_audit_trail_mobilenet_v3_small_20251027_102648.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/mobilenet_v3_small/corelation_all_mobilenet_v3_small.xlsx


python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/vgg16/sensitivity_audit_trail_vgg16_20251027_074803.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/vgg16/anova_all_vgg16.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/resnet50/sensitivity_audit_trail_resnet50_20251027_081738.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/resnet50/anova_all_resnet50.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/inception_v3/sensitivity_audit_trail_inception_v3_20251027_091150.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/inception_v3/anova_all_inception_v3.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/mobilenet_v3_large/sensitivity_audit_trail_mobilenet_v3_large_20251027_110826.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/mobilenet_v3_large/anova_all_mobilenet_v3_large.xlsx


python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_config_s_deer_all_OK/mobilenet_v3_small/sensitivity_audit_trail_mobilenet_v3_small_20251027_102648.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_all/mobilenet_v3_small/anova_all_mobilenet_v3_small.xlsx


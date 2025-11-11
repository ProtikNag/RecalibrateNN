#########################################################


mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/vgg16
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/resnet50
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/inception_v3
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/mobilenet_v3_small
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/mobilenet_v3_large

if [ "$run_corel" -eq 1 ]; then

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/vgg16/sensitivity_audit_trail_vgg16_20251102_010745.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/vgg16/corelation_bg_vgg16.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/resnet50/sensitivity_audit_trail_resnet50_20251102_024246.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/resnet50/corelation_bg_resnet50.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/inception_v3/sensitivity_audit_trail_inception_v3_20251104_000445.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/inception_v3/corelation_bg_inception_v3.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/mobilenet_v3_large/sensitivity_audit_trail_mobilenet_v3_large_20251102_020214.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/mobilenet_v3_large/corelation_bg_mobilenet_v3_large.xlsx


python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/mobilenet_v3_small/sensitivity_audit_trail_mobilenet_v3_small_20251102_012958.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/mobilenet_v3_small/corelation_bg_mobilenet_v3_small.xlsx
fi

if [ "$run_anova" -eq 1 ]; then

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/vgg16/sensitivity_audit_trail_vgg16_20251102_010745.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/vgg16/anova_bg_vgg16.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/resnet50/sensitivity_audit_trail_resnet50_20251102_024246.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/resnet50/anova_bg_resnet50.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/inception_v3/sensitivity_audit_trail_inception_v3_20251104_000445.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/inception_v3/anova_bg_inception_v3.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/mobilenet_v3_large/sensitivity_audit_trail_mobilenet_v3_large_20251102_020214.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/mobilenet_v3_large/anova_bg_mobilenet_v3_large.xlsx


python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_full_bg_OK/mobilenet_v3_small/sensitivity_audit_trail_mobilenet_v3_small_20251102_012958.csv  /mnt/sdb2/sensitivity_analysis_paper/corelation_bg/mobilenet_v3_small/anova_bg_mobilenet_v3_small.xlsx

fi

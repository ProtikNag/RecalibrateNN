#########################################################

mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/vgg16
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/resnet50
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/inception_v3
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/mobilenet_v3_small
mkdir -p /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/mobilenet_v3_large

if [ "$run_corel" -eq 1 ]; then
python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/vgg16/sensitivity_audit_trail_vgg16_20251026_200650.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/vgg16/corelation_leg_vgg16.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/resnet50/sensitivity_audit_trail_resnet50_20251026_203245.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/resnet50/corelation_leg_resnet50.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/inception_v3/sensitivity_audit_trail_inception_v3_20251026_212251.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/inception_v3/corelation_leg_inception_v3.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/mobilenet_v3_large/sensitivity_audit_trail_mobilenet_v3_large_20251026_230755.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/mobilenet_v3_large/corelation_leg_mobilenet_v3_large.xlsx

python corelation.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/mobilenet_v3_small/sensitivity_audit_trail_mobilenet_v3_small_20251026_222812.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/mobilenet_v3_small/corelation_leg_mobilenet_v3_small.xlsx

fi

if [ "$run_anova" -eq 1 ]; then

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/vgg16/sensitivity_audit_trail_vgg16_20251026_200650.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/vgg16/anova_leg_vgg16.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/resnet50/sensitivity_audit_trail_resnet50_20251026_203245.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/resnet50/anova_leg_resnet50.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/inception_v3/sensitivity_audit_trail_inception_v3_20251026_212251.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/inception_v3/anova_leg_inception_v3.xlsx

python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/mobilenet_v3_large/sensitivity_audit_trail_mobilenet_v3_large_20251026_230755.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/mobilenet_v3_large/anova_leg_mobilenet_v3_large.xlsx


python anova.py /mnt/sdb2/sensitivity_analysis_paper/sensitivity_analysis_leg/mobilenet_v3_small/sensitivity_audit_trail_mobilenet_v3_small_20251026_222812.csv /mnt/sdb2/sensitivity_analysis_paper/corelation_leg/mobilenet_v3_small/anova_leg_mobilenet_v3_small.xlsx

fi
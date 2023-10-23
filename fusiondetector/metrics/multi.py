import subprocess

# Define the command template
# command_template = "python pascalvoc.py --gt /home/ec2-user/dev/data/logo05/annotations/gts_preds/gts --det /home/ec2-user/dev/data/logo05/annotations/gts_preds/preds_yolov8s_t14 -t {} -conf 0.15"
command_template = "python pascalvoc.py --gt /home/ec2-user/dev/data/widerface/gts_preds/gts --det /home/ec2-user/dev/data/widerface/gts_preds/preds_yolov8s_t14  -t {} -conf 0.12" 

# Loop through different values of -t
for threshold in range(50, 100, 5):
    threshold_value = threshold / 100  # Convert to decimal
    command = command_template.format(threshold_value)
    subprocess.run(command, shell=True)
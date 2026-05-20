"""
Script to evaluate a YOLO Model with conventional inference.

Computing: Box-P, Box-R, Box-F1, mAP50, mAP50-95, custom_dr, custom_fpr

"""

# Import packages
from ultralytics import YOLO
import polars as pl
from custom_evaluation_functions import detection_rate, false_positive_rate
import subprocess
import glob

# Set these variables yourself to indicate which model should be tested (and with what inference parameters)
version = "26"
exp = "SF_s"
conf = 0.134
iou = 0.1

# Model is created
model_name = f"YOLOv{version}/{exp}"
weights_path = f"/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/{model_name}_train/weights/best.pt"
model = YOLO(weights_path)

# Model is evaluated
output_folder = f"/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/YOLOv{version}/evaluation_runs"


"""------------------------------------- Compute Precision, Recall, F1 and mAP scores ----------------------------------------"""
configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/yolo_config.yaml' # datapath 
print("Running default evaluation")
# Run evaluation
og_results = model.val(data=configuration, split='test', project=output_folder, name=exp, conf=conf, iou=iou, visualize=True)
print("Completed default evaluation")
for path in glob.glob("/tmp/pymp*"):
            subprocess.run(['rm', '-r', path], capture_output=True)
results_df = og_results.to_df()
print("Intermediate results save...")
results_df.write_csv(output_folder+f"/{exp}_evaluation_metrics.csv")


"""--------------------------------------------- Compute Detection Rate score ------------------------------------------------"""
print("Running custom DR evaluation")
custom_configuration = '/wecare/home/lotte/Thesis/CODE/ETZ/YOLO/yolo_custom_config.yaml' 
inputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/test_singulars" # datapath
# Run evaluation
custom_dr = detection_rate(model=model, version=version, exp=exp, custom_configuration=custom_configuration, conf=conf, iou=iou,
                            inputpath=inputpath, output_folder=output_folder, save=True, plots=True)
print("Completed custom DR evaluation:", custom_dr)
for path in glob.glob("/tmp/pymp*"):
            subprocess.run(['rm', '-r', path], capture_output=True)
results_df = results_df.with_columns(pl.Series("custom_dr", [custom_dr]))
print("Intermediate results save...")
results_df.write_csv(output_folder+f"/{exp}_evaluation_metrics.csv")


"""--------------------------------------------- Compute False Positive Rate -------------------------------------------------"""
print("Running custom FPR evaluation")
inputpath = "/wecare/home/lotte/Thesis/DATA/ETZ/negative_slices" # datapath
# Run evaluation
false_positive_images, custom_fpr = false_positive_rate(model=model, image_inputpath=inputpath, conf=conf, iou=iou)
print("Completed custom FPR evaluation", custom_fpr)
for path in glob.glob("/tmp/pymp*"):
            subprocess.run(['rm', '-r', path], capture_output=True)
results_df = results_df.with_columns(pl.Series("custom_fpr", [custom_fpr]))



# Save all evaluation results:
print("Saving all evaluation metrics")
results_df.write_csv(output_folder+f"/{exp}_evaluation_metrics.csv")
print("Saved evaluation metrics to:", output_folder+f"/{exp}_evaluation_metrics.csv")
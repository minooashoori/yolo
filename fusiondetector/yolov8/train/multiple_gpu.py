import sys
import os
from ultralytics import YOLO, settings
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_yolo(config):
    model = YOLO('yolov8n.yaml')  # build a new model from YAML

    results = model.train(data="/home/ubuntu/ctx-logoface-detector/fusiondetector/yolov8/datasets/minoo.yaml",
                          epochs=config["epochs"],
                          batch=config["batch"],
                          imgsz=416,
                          mosaic=config["mosaic"],
                          close_mosaic=10,
                          device=[0, 1, 2, 3],
                          workers=32,
                          optimizer=config["optimizer"],
                          save_period=1,
                          pretrained=True,
                          lr0=config["lr0"],
                          save_conf=True,
                          save_crop=True,
                          augment=True,
                          half=True,
                          verbose=True,
                          save_json=False,
                          visualize=True,
                          degrees=5.0,
                          flipud=0.005,
                         )
    
    # Report the metric to Ray Tune
    tune.report(mAP50_B=results.results_dict['metrics/mAP50(B)'])

config = {
    "epochs": tune.choice([2, 5, 10]),
    "batch": tune.choice([64, 128, 256]),
    "mosaic": tune.choice([True, False]),
    "optimizer": tune.choice(["SGD", "Adam"]),
    "lr0": tune.loguniform(1e-5, 1e-2),
}

# Define the scheduler
scheduler = ASHAScheduler(
    metric="mAP50_B",  # Using the metric from results
    mode="max",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

analysis = tune.run(
    train_yolo,
    resources_per_trial={"gpu": 4},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=tune.CLIReporter(metric_columns=["mAP50_B"])
)

print("Best hyperparameters found were: ", analysis.best_config)

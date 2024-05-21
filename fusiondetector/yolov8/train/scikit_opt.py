# from ultralytics import YOLO

# # Initialize the YOLO model
# model = YOLO('yolov8n.yaml')

# # Tune hyperparameters on COCO8 for 30 epochs
# model.tune(data="/home/ubuntu/ctx-logoface-detector/fusiondetector/yolov8/datasets/minoo.yaml",
#            epochs=5, iterations=10, optimizer='AdamW', plots=False, save=False, val=False)
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from ultralytics import YOLO
import numpy as np

# Define the search space
search_space = [
    # Categorical([20, 30, 40, 50], name='epochs'),
    Categorical([16, 32, 64, 128], name='batch'),
    Real(0.00001, 0.001, "log-uniform", name='lr0'),
    Categorical(['SGD', 'Adam'], name='optimizer'),
    Categorical([5, 7, 9, 11, 13, 15], name='box'),

]

# Define the objective function
@use_named_args(search_space)
def objective(**params):
    model = YOLO('yolov8n.yaml')
  
  
    results = model.train(
        data="/home/ubuntu/ctx-logoface-detector/fusiondetector/yolov8/datasets/minoo.yaml",
        imgsz=416,
        mosaic=False,
        close_mosaic=10, 
        workers=32,
        epochs = 40,
        batch = params['batch'],
        optimizer=params['optimizer'],
        lr0=params['lr0'],
        box=params['box']
    )
    if results is None:
      pass
    else:
      loss1 = results.results_dict['metrics/mAP50-95(B)']
    return -loss1  # Negate if higher metric values are better

  
# Run the optimization
result = gp_minimize(objective, search_space, n_calls=30, random_state=0)
print(result)
print(f"Best hyperparameters: {result.x}")

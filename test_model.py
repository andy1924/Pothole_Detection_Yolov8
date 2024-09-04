import os
from ultralytics import YOLO

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)
metrics = model.val()  # assumes `model` has been loaded

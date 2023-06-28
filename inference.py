import torch
import cv2
import numpy as np
from pathlib import Path
import easyocr
# Model
model_path = Path("yolov5/runs/exp/weights/best.pt") 
img_path = Path("test_images/cars.jpg")   
device = torch.device('cuda')

model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=True)
model = model.to(device)
image = cv2.imread(img_path)
image = cv2.resize(image, (640,640), interpolation = cv2.INTER_AREA)
reader = easyocr.Reader(['en','ru'], gpu = True)
# Inference
output = model(image)

# Results
result = np.array(output.pandas().xyxy[0])
for i in result:
    p1 = (int(i[0]),int(i[1]))
    p2 = (int(i[2]),int(i[3]))
    res = reader.readtext(image[i[1]:i[3],i[0]:i[2]],detail = 0)
    print(''.join(res))

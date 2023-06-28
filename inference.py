import torch
import cv2
import numpy as np
from pathlib import Path
import easyocr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_file_name', type=str)
parser.add_argument('--result_file', type=str)
args = parser.parse_args()
# Model
model_path = Path("yolov5/runs/exp/weights/best.pt") 
img_path = Path("test_images/cars.jpg")   
device = torch.device('cuda')

model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=True)
model = model.to(device)

reader = easyocr.Reader(['en','ru'], gpu = True)

image = cv2.imread(args.img_file_name)

# Inference
output = model(image)
result = np.array(output.pandas().xyxy[0])
lst_results = []
for i in result:
    p1 = (int(i[0]),int(i[1]))
    p2 = (int(i[2]),int(i[3]))
    res = reader.readtext(image[i[1]:i[3],i[0]:i[2]],detail = 0)
    print(''.join(res))
    lst_results.append(''.join(res))

with open(args.result_file, 'w') as fp:
    for row in lst_results:
        fp.write(f"{row}\n")


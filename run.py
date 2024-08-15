import torch
import pandas as pd
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp8/weights/last.pt')

cap = cv2.VideoCapture('E://jupyternotebook//robot2.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (1550,900), fx=0, fy=0, interpolation=cv2.INTER_AREA)
    results = model(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
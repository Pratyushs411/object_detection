from ultralytics import YOLO
import cv2
model = YOLO('../yolo-weights/yolov8m.pt')
results = model("images/1.jpg",show=True)
results1 = model("images/3.jpg",show=True)
cv2.waitKey(0)
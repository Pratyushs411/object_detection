from ultralytics import YOLO
import cv2
import cvzone
import math
cap = cv2.VideoCapture("../videos/video_4.mp4")
# cap.set(3, 1280)
# cap.set(4,720)
model = YOLO('../yolo-weights/yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "com", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella","handbag","tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "sofa", "potted plant", "bed","dining table", "toilet",
            "tv monitor", "Laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
              "vase", "scissors", "teddy bear", "hair drier", "toothbrush","key",]
while True:
    success, img=cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1), int(y1), int(x2), int(y2)
            print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]}' f'{conf}',(max(0,x1),max(35,y1)),scale=1.5,thickness=2)

    cv2.imshow("image",img)
    cv2.waitKey(1)
import cv2
from ultralytics import YOLO
import numpy as np



mydevice = "mps"

cap =cv2.VideoCapture(1)

model = YOLO("saved_models/hat_best.pt")
print("Model loaded")

ret, frame = cap.read()

while ret:
    

    results = model(frame, device=mydevice)

    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    print(bboxes)
    
    for cls,bbox in zip(classes, bboxes):
        x1, y1, x2, y2 = bbox
        if cls == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, model.names[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
        
    
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()










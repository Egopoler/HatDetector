import cv2
from ultralytics import YOLO
import numpy as np
import torch
import time



def get_device():
    """
    Returns the device to be used for computations.

    Returns:
        str: The device to be used for computations. Possible values are "mps", "cuda", or "cpu".
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"



def connect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap
    print("Error: Could not open video")
    exit(0)


def start_detection(model, cap, mydevice="cpu"):
    
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





mydevice = get_device()

# I choose this video in data_videos folder for example, you can load your own video
# This video from this website "https://ru.freepik.com/free-video/portrait-shot-successful-farmer-wearing-hat_472639/16:9#fromView=search&page=1&position=6&uuid=2a15a57b-1da0-4493-80d0-bc54cf45df1c"
cap = connect_video("data_videos/hat_test_video_1.mp4")

model = YOLO("saved_models/hat_best.pt") 

print("Model loaded")

# To check if model is loaded correctly and video is working
# time.sleep(2)

start_detection(model=model, cap=cap, mydevice=mydevice)




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


def get_first_available_camera(prior_camera=1,max_cameras=10):
    """try to open cameras from 0 to max_cameras-1 and return index of the first one that works."""
    cap = cv2.VideoCapture(prior_camera)
    if cap.isOpened():
        return prior_camera
    cap.release()

    for camera_index in range(max_cameras):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            return camera_index
        cap.release()
    return None

def connect_webcam(prior_camera=0,max_cameras=10):
    """
    Connects to the first available webcam and returns a VideoCapture object.

    Args:
        prior_camera (int, optional): The index of the camera to use. Defaults to 0.
        max_cameras (int, optional): The maximum number of available cameras. Defaults to 10.
    Returns:
        cv2.VideoCapture: A VideoCapture object representing the connected webcam.

    Raises:
        SystemExit: If no available webcam is found.

    """
    camera_index = get_first_available_camera(prior_camera=prior_camera,max_cameras=max_cameras)

    if camera_index is not None:
        print(f"Using camera with index: {camera_index}")
        cap = cv2.VideoCapture(camera_index)
        
        # try to open the camera
        if cap.isOpened():  
            cap = cv2.VideoCapture(camera_index)
            return cap
    print("Error: Could not open webcamera")
    exit(0)



def start_detection(model, cap, mydevice="cpu"):
    """
    Starts the detection process using a given model and camera.

    Args:
        model (object): The model used for object detection.
        cap (cv2.VideoCapture): The video capture object representing the camera.
        mydevice (str, optional): The device to be used for computations. Defaults to "cpu".

    Returns:
        None

    Description:
        This function continuously reads frames from the camera and performs object detection using the provided model.
        It displays the detected objects on the frame and allows the user to exit by pressing the 'Esc' key.

        The function uses the following steps:
        1. Reads the first frame from the camera.
        2. Loops until there are no more frames.
        3. Performs object detection on the current frame using the provided model.
        4. Retrieves the bounding boxes and class labels of the detected objects.
        5. Draws rectangles around the detected objects with the class label on the frame.
        6. Displays the frame with the detected objects.
        7. Waits for a key press and breaks the loop if the 'Esc' key is pressed.
        8. Releases the camera and destroys all windows.

    Note:
        - The function assumes that the provided model has a method called "names" that returns the class labels.   
    """
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

# In my case I choose camera 1, but you can choose any camera. If you dont know the index of the camera you can delete argument prior_camera=1
cap = connect_webcam(prior_camera=1)

model = YOLO("saved_models/hat_best.pt") 

print("Model loaded")

# To check if model is loaded correctly and camera is working
# time.sleep(2)

start_detection(model=model, cap=cap, mydevice=mydevice)




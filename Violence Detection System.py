import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ultralytics import YOLO
import matplotlib.pyplot as plt
# from picamera2 import Picamera2
# import time

# Step 1: YOLO Object Detection
def detect_objects(frame, yolo_model, classes):
    results = yolo_model(frame)
    for result in results:
        bboxes = result.boxes
        b=bboxes.xyxy.cpu().numpy()
        boxes=b.astype(int)
        cls=bboxes.cls
        class_indices=cls.cpu().numpy()
        labels = [result.names[i] for i in class_indices]
    return boxes, labels

# Step 2: Image Classification using CNN
def classify_frame(frame, model):
    # Preprocess the frame (resize, normalize, etc.)
    frame = cv2.resize(frame, (224, 224))  # Assuming input size of the CNN model
    frame = frame / 255.0  # Normalize pixel values to [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    # Perform violence classification using the CNN model
    prediction = model.predict(frame)
    if prediction[0][0] > 0.5:
        return "Violent"
    else:
        return "Non-Violent"
    
def imshow(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(frame)
    plt.axis('off')  # Turn off axis
    plt.show()
    
def save_processed_frame(frame, save_path):
    cv2.imwrite(save_path, frame)
    return save_path

def process_video():
    # Load YOLO model
    yolo_model = YOLO('yolov8n.pt')  # You can specify yolov5s, yolov5m, yolov5l, or yolov5x
    # Load class labels (coco.names)
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Load CNN model for violence classification
    cnn_model = load_model("D:/lab_last/model.h5")
    
    # Open the webcam
      # Use index 0 for the primary webcam
    # cam = Picamera2()
    # cam.preview_configuration.main.size = (640, 360)
    # cam.preview_configuration.main.format = "RGB888"
    # cam.preview_configuration.controls.FrameRate = 30
    # cam.preview_configuration.align()
    # cam.configure("preview")
    # cam.start()
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the webcam
        ret, frame = video_capture.read()
        # frame = cam.capture_array()
        if not ret:
            break  # Break the loop if there are no more frames
        
        # Step 1: YOLO Object Detection
        detected_boxes, detected_labels = detect_objects(frame, yolo_model, classes)
        
        # Check if YOLO detected humans, guns, or knives
        if any(label in ['person', 'gun', 'knife'] for label in detected_labels):
            # Step 2: Image Classification using CNN
            for bbox, label in zip(detected_boxes, detected_labels):
                x, y, w, h = bbox
                object_frame = frame[y:y+h, x:x+w]
                predicted_class = classify_frame(object_frame, cnn_model)
                # Perform further processing based on the predicted class
                if predicted_class == 'Violent':
                    # Draw bounding box and label for violent object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Violent', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # Add your warning logic here
                    print("Warning: Violent Object Detected!")
                else:
                    # Draw bounding box and label for non-violent object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Non-Violent', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Add your non-violent object logic here
        
        # Display the processed frame
        cv2.imshow('Violence Detection', frame)
        
        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all windows
    # cam.stop
    # # vid.release()
    # cv.destroyAllWindows()
    video_capture.release()
    cv2.destroyAllWindows()

# Call the main function to process video from the webcam
process_video()

# Violence Detection using YOLO and CNN

This project implements violence detection in live video streams using the YOLO object detection algorithm and a Convolutional Neural Network (CNN) for image classification.

## Overview

Violence detection is an important application in video surveillance and security systems. This project aims to detect violent activities in live video streams in real-time using deep learning techniques.

The project consists of two main components:

1. **Object Detection using YOLO (You Only Look Once)**:
    - YOLO is used to detect objects of interest in each frame of the video stream. Specifically, we are interested in detecting humans, guns, and knives.

2. **Image Classification using CNN**:
    - A CNN model is employed to classify detected objects into violent or non-violent categories. The model is trained to recognize violent behaviors based on image features extracted from the detected objects.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- PyTorch (for YOLO object detection)
- Ultralytics (for YOLO object detection)

## File

1. **model-building.ipnyb**: This file used Train the CNN model with the dataset

2. **Violence Detection System.py**: This file has the code which contain the integrated part of yolo and cnn.


## Usage

1. Run the main script to start violence detection in live video streams:

    ```bash
    python Violence Detection System.py
    ```

2. The program will use your laptop's webcam to capture live video frames.

3. Detected violent objects will be highlighted in red rectangles with a label "Violent", while non-violent objects will be highlighted in green rectangles with a label "Non-Violent".

4. The program will display the processed video stream with violence detection results in real-time.

5. To change it t rasberrypi usage u can uncomment the code kine to activate rasberrypi cam and comment line to activate webcam.


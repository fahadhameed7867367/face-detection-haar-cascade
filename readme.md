# Classical Face Detection using Haar Cascades

## Overview
This repository contains a Python implementation of a classical face detection pipeline using OpenCV and Haar Cascade classifiers. It was developed as part of my undergraduate computer vision work.

## Pipeline Architecture
The script implements a standard classical detection pipeline:
1. **Grayscale Conversion**: Haar cascades require single-channel images.
2. **Histogram Equalization**: A basic contrast enhancement step used to mitigate poor illumination.
3. **Multi-scale Scanning**: The classifier evaluates every window at multiple scales (`detectMultiScale`). 
4. **Non-Maximum Suppression (NMS)**: Handled internally by OpenCV to merge overlapping detections.

## Documented Limitations
While this system works reliably on clean, frontal, well-lit faces, my evaluation documented significant failure modes:
- **Partial Occlusion**: The fixed template matching fails when structural features (e.g., eyes, nose) are occluded.
- **Profile Views**: The default cascade is trained on frontal faces and fails on side profiles.
- **Low Light**: Histogram equalization is insufficient to recover structure lost in extreme low-light conditions.

These failures motivated my interest in deep learning-based restoration, learned representations, and test-time adaptation.

## How to Run
1. Install OpenCV: `pip install opencv-python`
2. Run the script: `python face_detection.py --image path_to_your_image.jpg`
3. The output image with bounding boxes will be saved in the `output/` directory.
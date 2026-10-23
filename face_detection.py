import cv2
import os
import argparse

def detect_faces(image_path, output_dir="output"):
    """
    Detects faces in an image using OpenCV's Haar Cascade classifier.
    This pipeline demonstrates the classical approach: grayscale conversion, 
    histogram equalization, and exhaustive multi-scale window scanning.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the pre-trained Haar Cascade classifier for frontal faces
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Could not load Haar cascade file.")
        return

    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # 1. Convert to grayscale (Haar cascades require grayscale input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Apply Histogram Equalization
    # This helps improve contrast, which is a naive attempt to handle low-light conditions
    gray_eq = cv2.equalizeHist(gray)

    # 3. Exhaustive multi-scale window scanning
    # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
    # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"Found {len(faces)} face(s) in {image_path}")

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save the output image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"detected_{filename}")
    cv2.imwrite(output_path, img)
    print(f"Saved output to {output_path}")
    
    if len(faces) == 0:
        print("Note: No faces found. This pipeline is known to fail under low light, profile views, or partial occlusion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical Face Detection using Haar Cascades")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    detect_faces(args.image)
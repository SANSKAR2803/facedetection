from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8-Face model
model = YOLO("yolov8n-face.pt")  # We'll download this model

def detect_faces(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path")

    # Perform detection
    results = model(img)

    # Extract the first detected face (if any)
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]  # First face
        x, y, w, h = box.xywh[0].cpu().numpy()  # Get bounding box as [x_center, y_center, width, height]
        x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)  # Convert to top-left corner format

        # Crop the face
        face = img[y:y+h, x:x+w]
        return face
    else:
        raise ValueError("No faces detected in the image")

if __name__ == "__main__":
    # Test with a sample image (replace with your own image path)
    test_image = "datasets/known_faces/phaneendra.jpg"
    try:
        face = detect_faces(test_image)
        cv2.imwrite("detected_face.jpg", face)  # Save the cropped face
        print("Face detected and saved as 'detected_face.jpg'")
    except Exception as e:
        print(f"Error: {e}")
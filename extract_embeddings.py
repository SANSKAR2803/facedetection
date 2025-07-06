from ultralytics import YOLO
import cv2
import numpy as np
import onnxruntime as ort
import redis

# Load YOLO model for face detection
yolo_model = YOLO("yolov8n-face.pt")

# Path to the ArcFace ONNX model
arcface_model_path = r"C:\Users\phane\.insightface\models\buffalo_l\w600k_r50.onnx"

# Load ArcFace model with ONNX Runtime
session = ort.InferenceSession(arcface_model_path, providers=['CPUExecutionProvider'])

# Connect to Redis (running in Docker on localhost:6379)
r = redis.Redis(host='localhost', port=6379, db=0)

def detect_faces(image_path):
    """Detect faces in an image using YOLOv8 and return the first detected face."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path")
    results = yolo_model(img)
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x, y, w, h = box.xywh[0].cpu().numpy()
        x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
        face = img[y:y+h, x:x+w]
        return face
    else:
        raise ValueError("No faces detected")

def extract_embedding(face):
    """Extract embedding from a cropped face image using the ArcFace model."""
    face_resized = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_input = (face_rgb.astype(np.float32) - 127.5) / 127.5
    face_input = face_input.transpose(2, 0, 1)[np.newaxis, :]
    embedding = session.run(None, {"input.1": face_input})[0]
    return embedding.squeeze()

def store_embedding(user_id, embedding):
    """Store the embedding in Redis with a user ID."""
    # Convert embedding to bytes for Redis storage
    embedding_bytes = embedding.tobytes()
    r.set(f"embedding:{user_id}", embedding_bytes)
    print(f"Stored embedding for user_id: {user_id}")

def retrieve_embedding(user_id):
    """Retrieve an embedding from Redis and convert back to numpy array."""
    embedding_bytes = r.get(f"embedding:{user_id}")
    if embedding_bytes:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        return embedding
    else:
        raise ValueError(f"No embedding found for user_id: {user_id}")

if __name__ == "__main__":
    test_image = "datasets/known_faces/phaneendra.jpg"
    test_user_id = "user1"  # Example user ID
    try:
        # Detect and save the face
        face = detect_faces(test_image)
        cv2.imwrite("detected_face.jpg", face)
        print("Face detected and saved as 'detected_face.jpg'")

        # Extract embedding
        embedding = extract_embedding(face)
        print("Embedding shape:", embedding.shape)
        print("Embedding sample:", embedding[:5])

        # Store in Redis
        store_embedding(test_user_id, embedding)

        # Retrieve and verify
        retrieved_embedding = retrieve_embedding(test_user_id)
        print("Retrieved embedding shape:", retrieved_embedding.shape)
        print("Retrieved embedding sample:", retrieved_embedding[:5])
        # Check if they match
        if np.array_equal(embedding, retrieved_embedding):
            print("Embedding stored and retrieved successfully!")
        else:
            print("Error: Retrieved embedding doesn’t match original.")

    except Exception as e:
        print(f"Error: {e}")
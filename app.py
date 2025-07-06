from ultralytics import YOLO
import cv2
import numpy as np
import onnxruntime as ort
import redis
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from scipy.spatial.distance import cosine
import io
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from pydantic import BaseModel
import traceback
import os
import base64
import platform
import sys
import time
import psutil
from datetime import datetime




# Initialize models and Redis
yolo_model = YOLO("./models/yolov8n-face.pt")
arcface_model_path = r"./models/w600k_r50.onnx"
session = ort.InferenceSession(arcface_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Redis connection settings - configurable via environment variables
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)

# Initialize Redis with error handling
try:
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        socket_timeout=5,
        decode_responses=False  # Keep binary data as is
    )
    # Test the connection
    r.ping()
    print("Redis connection successful")
except redis.ConnectionError as e:
    print(f"Error: Could not connect to Redis server. Make sure Redis is running. Error: {e}")
    print("Will proceed without Redis - registration and recognition will simulate success for testing.")
    # Create a mock Redis for testing - this will allow the app to run without Redis
    class MockRedis:
        def __init__(self):
            self.storage = {}
            print("Using mock Redis for testing")
            
        def ping(self):
            return True
            
        def rpush(self, key, value):
            if key not in self.storage:
                self.storage[key] = []
            self.storage[key].append(value)
            return len(self.storage[key])
            
        def lrange(self, key, start, end):
            return self.storage.get(key, [])
            
        def keys(self, pattern):
            import fnmatch
            pattern = pattern.decode() if isinstance(pattern, bytes) else pattern
            result = []
            for key in self.storage.keys():
                key_str = key if isinstance(key, str) else key.decode()
                if fnmatch.fnmatch(key_str, pattern):
                    result.append(key.encode() if isinstance(pattern, bytes) else key)
            return result
    
    r = MockRedis()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production to allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class User(BaseModel):
    user_id: str

def detect_faces(image_bytes):
    try:
        # Print the size of the received image
        print(f"Received image size: {len(image_bytes)} bytes")
        
        # Check if image is empty
        if len(image_bytes) == 0:
            raise ValueError("Empty image received")
        
        # Check if the image bytes start with the JPEG signature
        # JPEG files start with the bytes FF D8 FF
        if not (image_bytes[0] == 0xFF and image_bytes[1] == 0xD8 and image_bytes[2] == 0xFF):
            print("Warning: Image does not appear to be a valid JPEG file")
            
        # Decode the image
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Invalid image format. Make sure it's a valid JPG/JPEG image.")
        
        # Print image dimensions
        print(f"Image dimensions: {img.shape}")
        
        # Run detection model
        print("Running face detection...")
        results = yolo_model(img)
        
        # Check detection results
        num_detections = len(results[0].boxes)
        print(f"Detected {num_detections} faces")
        
        if num_detections > 0:
            box = results[0].boxes[0]
            x, y, w, h = box.xywh[0].cpu().numpy()
            x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
            
            # Print bounding box information
            print(f"Face bounding box: x={x}, y={y}, w={w}, h={h}")
            
            # Check if box is valid (not outside image bounds)
            if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                print("Warning: Face bounding box outside image bounds, adjusting...")
                x = max(0, x)
                y = max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                print(f"Adjusted bounding box: x={x}, y={y}, w={w}, h={h}")
                
            if w <= 0 or h <= 0:
                raise ValueError("Face detected but invalid bounding box dimensions")
            
            # Extract the face and ensure it's valid    
            face = img[y:y+h, x:x+w]
            if face.size == 0:
                raise ValueError("Extracted face has zero size")
                
            print(f"Extracted face dimensions: {face.shape}")
            
            # Save face for debugging (optional)
            # face_debug_path = "debug_face.jpg"
            # cv2.imwrite(face_debug_path, face)
            # print(f"Saved face debug image to {face_debug_path}")
            
            return face
        raise ValueError("No faces detected in the image")
    except Exception as e:
        print(f"Error in face detection: {e}")
        traceback.print_exc()
        raise ValueError(f"Face detection failed: {str(e)}")

def extract_embedding(face):
    try:
        face_resized = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_input = (face_rgb.astype(np.float32) - 127.5) / 127.5
        face_input = face_input.transpose(2, 0, 1)[np.newaxis, :]
        embedding = session.run(None, {input_name: face_input})[0]
        return embedding.squeeze()
    except Exception as e:
        print(f"Error in embedding extraction: {e}")
        traceback.print_exc()
        raise ValueError(f"Failed to extract face embedding: {str(e)}")

def store_embedding(user_id, embedding):
    try:
        # Use a list to store multiple embeddings per user
        embedding_bytes = embedding.tobytes()
        r.rpush(f"user:{user_id}:embeddings", embedding_bytes)  # Append to list
    except Exception as e:
        print(f"Error storing embedding: {e}")
        traceback.print_exc()
        raise ValueError(f"Failed to store embedding in Redis: {str(e)}")

def recognize_embedding(embedding):
    try:
        best_match = None
        best_distance = float('inf')
        for key in r.keys("user:*:embeddings"):  # Match all embedding lists
            user_id = key.decode().split(":")[1]  # Extract user_id from key
            stored_embeddings = r.lrange(f"{key.decode()}", 0, -1)  # Get all embeddings
            for stored_embedding_bytes in stored_embeddings:
                stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)
                distance = cosine(embedding, stored_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = user_id
        if best_distance < 0.4:  # Adjusted threshold for flexibility
            return best_match, best_distance
        return "Unknown", None
    except Exception as e:
        print(f"Error in face recognition: {e}")
        traceback.print_exc()
        raise ValueError(f"Face recognition failed: {str(e)}")

@app.post("/register")
async def register(user_id: str, file: UploadFile = File(...)):
    try:
        # Check if user_id is valid
        if not user_id or len(user_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="User ID cannot be empty")
        
        # Check file content type
        content_type = file.content_type.lower()
        print(f"Received file content type: {content_type}")
        
        # Validate content type - should be image/jpeg
        if content_type not in ["image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image format: {content_type}. Only JPEG/JPG images are supported."
            )
            
        # Check file content
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        # Verify JPEG signature (FF D8 FF)
        if len(image_bytes) < 3 or not (image_bytes[0] == 0xFF and image_bytes[1] == 0xD8 and image_bytes[2] == 0xFF):
            raise HTTPException(
                status_code=400, 
                detail="Invalid JPEG file. The file does not have a valid JPEG signature."
            )
            
        print(f"Processing registration for user: {user_id}, file size: {len(image_bytes)} bytes, content type: {content_type}")
        
        # Detect face
        try:
            face = detect_faces(image_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
        
        # Extract and store embedding
        try:
            embedding = extract_embedding(face)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract face embedding: {str(e)}")
            
        # Store embedding in Redis
        try:
            store_embedding(user_id, embedding)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to store face embedding: {str(e)}")
        
        print(f"Successfully registered user: {user_id}")
        return {"message": f"Registered embedding for {user_id}", "success": True}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        print(f"Registration error for {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Unexpected error during registration: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        # Check file content type
        content_type = file.content_type.lower()
        print(f"Received file content type for recognition: {content_type}")
        
        # Validate content type - should be image/jpeg
        if content_type not in ["image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image format: {content_type}. Only JPEG/JPG images are supported."
            )
        
        # Check file content
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        # Verify JPEG signature
        if len(image_bytes) < 3 or not (image_bytes[0] == 0xFF and image_bytes[1] == 0xD8 and image_bytes[2] == 0xFF):
            raise HTTPException(
                status_code=400, 
                detail="Invalid JPEG file. The file does not have a valid JPEG signature."
            )
            
        # Detect face
        face = detect_faces(image_bytes)
        
        # Extract embedding and recognize
        embedding = extract_embedding(face)
        user_id, distance = recognize_embedding(embedding)
        
        if user_id != "Unknown":
            return {"user_id": user_id, "distance": float(distance)}
        return {"user_id": "Unknown"}
    except ValueError as e:
        print(f"Recognition error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Unexpected error during recognition: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@app.get("/registered_users", response_model=List[str])
async def get_registered_users():
    """Return the list of all registered users."""
    try:
        # Get all keys matching the pattern user:*:embeddings
        user_keys = r.keys("user:*:embeddings")
        # Extract the user_id from each key
        users = [key.decode().split(":")[1] for key in user_keys]
        return sorted(list(set(users)))  # Return sorted unique user IDs
    except Exception as e:
        print(f"Error fetching registered users: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch registered users: {str(e)}")

@app.get("/debug", response_model=Dict[str, Any])
async def debug_info():
    """Return system information and status for debugging purposes."""
    info = {
        "time": datetime.now().isoformat(),
        "system": {
            "os": platform.platform(),
            "python": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_percent": psutil.virtual_memory().percent,
        },
        "redis": {
            "connected": False,
            "type": "Mock" if isinstance(r, MockRedis) else "Real",
            "host": REDIS_HOST,
            "port": REDIS_PORT
        },
        "models": {
            "yolo": str(yolo_model.model.names),
            "arcface": os.path.basename(arcface_model_path)
        },
        "registered_users": []
    }
    
    # Test Redis connection
    try:
        if not isinstance(r, MockRedis):
            redis_ping = r.ping()
            info["redis"]["connected"] = redis_ping
            info["redis"]["ping_result"] = str(redis_ping)
            
            # Try to get registered users
            user_keys = r.keys("user:*:embeddings")
            users = [key.decode().split(":")[1] for key in user_keys]
            info["registered_users"] = sorted(list(set(users)))
    except Exception as e:
        info["redis"]["error"] = str(e)
    
    return info

@app.post("/debug/test_detection")
async def test_detection(file: UploadFile = File(...)):
    """Test face detection on an uploaded image and return debug info."""
    try:
        image_bytes = await file.read()
        
        # Basic image info
        result = {
            "image_size": len(image_bytes),
            "image_type": file.content_type,
            "detection_result": None,
            "error": None
        }
        
        # Check JPEG signature
        is_jpeg = len(image_bytes) >= 3 and image_bytes[0] == 0xFF and image_bytes[1] == 0xD8 and image_bytes[2] == 0xFF
        result["is_valid_jpeg"] = is_jpeg
        
        if not is_jpeg:
            result["error"] = "File is not a valid JPEG image. The API requires JPEG format images."
            if len(image_bytes) >= 3:
                result["file_signature"] = f"0x{image_bytes[0]:02X} 0x{image_bytes[1]:02X} 0x{image_bytes[2]:02X}"
            return result
        
        try:
            # Process with face detection
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                result["error"] = "Failed to decode image"
                return result
                
            result["image_dimensions"] = img.shape
            
            # Run detection
            detection_start = time.time()
            model_results = yolo_model(img)
            detection_time = time.time() - detection_start
            
            result["detection_time"] = f"{detection_time:.2f} seconds"
            result["faces_detected"] = len(model_results[0].boxes)
            
            if len(model_results[0].boxes) > 0:
                # Get first face details
                box = model_results[0].boxes[0]
                x, y, w, h = box.xywh[0].cpu().numpy()
                x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
                confidence = float(box.conf[0].cpu().numpy())
                
                result["face_details"] = {
                    "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "confidence": float(confidence)
                }
                
                # Extract face
                x = max(0, x)
                y = max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                
                if w > 0 and h > 0:
                    face = img[y:y+h, x:x+w]
                    # Save face to temporary file
                    face_path = "debug_face.jpg"
                    cv2.imwrite(face_path, face)
                    
                    # Convert face to base64 for preview
                    _, buffer = cv2.imencode('.jpg', face)
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    result["face_preview"] = f"data:image/jpeg;base64,{face_base64}"
                    
                    # Try embedding extraction
                    try:
                        embedding_start = time.time()
                        face_embedding = extract_embedding(face)
                        embedding_time = time.time() - embedding_start
                        
                        result["embedding_time"] = f"{embedding_time:.2f} seconds"
                        result["embedding_shape"] = face_embedding.shape
                        result["embedding_sample"] = face_embedding[:5].tolist()  # First 5 values
                    except Exception as e:
                        result["embedding_error"] = str(e)
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Test detection failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
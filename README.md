# Face Recognition Attendance System

A web-based attendance system that uses face recognition to track attendance of registered individuals.

## Features

- **Register**: Register new individuals in the system by capturing their face
- **Recognize**: Real-time face recognition from webcam feed
- **Attendance Tracking**: Monitor who is present and who is absent
- **Attendance Report**: View a summary of attendance status

## Tech Stack

- **Backend**: FastAPI, Redis, Ultralytics YOLO, ONNXRuntime
- **Frontend**: HTML, CSS, JavaScript
- **Face Recognition**: YOLOv8 (face detection) + ArcFace (face embedding)

## Setup and Installation

### Prerequisites

- Python 3.8+
- Redis server
- Node.js (optional, for development server)

### Backend Setup

1. Install required Python packages:

   ```
   pip install ultralytics opencv-python numpy onnxruntime redis fastapi python-multipart uvicorn scipy
   ```

2. Make sure Redis server is running:

   ```
   redis-server
   ```

3. Run the FastAPI backend:
   ```
   python app.py
   ```

### Frontend Setup

1. Simply open `index.html` in a web browser, or serve it with a basic HTTP server:

   ```
   npx http-server -p 3000
   ```

2. Access the application at `http://localhost:3000`

## How to Use

### Registration

1. Navigate to the "Register" tab
2. Enter a name or ID for the person
3. Click "Capture and Register" to take a photo and register the face

### Attendance Tracking

1. Navigate to the "Recognize" tab
2. Click "Start Recognition" to begin monitoring
3. As people appear in the camera feed, they will be recognized and marked as present
4. View the "Attendance" tab to see the status of all registered individuals

## API Endpoints

- `POST /register` - Register a new face with a user ID
- `POST /recognize` - Recognize a face from an image
- `GET /registered_users` - Get a list of all registered users

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.

## License

MIT License

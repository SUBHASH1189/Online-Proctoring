# Online Proctoring System

This project is an online proctoring platform designed to monitor and ensure the integrity of online examinations using advanced computer vision and audio analysis techniques.

## Features
- **Facial Detection**: Detects and tracks faces to ensure the presence of the candidate.
- **Eye Tracking**: Monitors eye movement to detect suspicious behavior.
- **Head Pose Estimation**: Identifies head movements to prevent cheating.
- **Object Detection**: Detects unauthorized objects using YOLO models.communication.
- **User Authentication**: Secure login and user management.
- **Admin & Proctor Dashboards**: Manage users, monitor exams, and review logs.
- **Database Integration**: Stores user data, logs, and exam results using SQLAlchemy (PostgreSQL/MySQL).

## Project Structure
```
proctoring/
    audio_detection.py
    database.py
    eye_tracker.py
    facial_detection.py
    head_pose_estimation.py
    main.py
    models.py
    object_detection.py
    routers/
        admin.py
        auth.py
        proctor.py
        users.py
    object_detection_model/
        config/
        objectLabels/
        weights/
    shape_predictor_model/
    static/
    templates/
logs/
requirements.txt
```

## Setup Instructions
1. **Clone the repository**
2. **Create and activate a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Configure the database**
   - Update `proctoring/database.py` for PostgreSQL or MySQL connection.
   - Create the database `Proctoring` in your DBMS.
5. **Run the application**
   ```powershell
   uvicorn proctoring.app:main --reload
   ```
6. **Access the app**
   - Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Requirements
- Python 3.8+
- PostgreSQL or MySQL
- SQLAlchemy
- FastAPI
- OpenCV
- PyTorch or TensorFlow (for object detection)
- Other dependencies in `requirements.txt`

## Model Files
- YOLO config and weights: `proctoring/object_detection_model/`
- Dlib shape predictor: `proctoring/shape_predictor_model/`

## Usage
- Start the server and log in as admin, proctor, or user.
- Monitor exams in real-time.
- Review logs and flagged events.

## License
This project is for educational purposes. Please check individual file headers for license details.

## Authors
- Vikkurthi Sai Subhash

## Acknowledgements
- OpenCV
- Dlib
- YOLO
- FastAPI
- SQLAlchemy

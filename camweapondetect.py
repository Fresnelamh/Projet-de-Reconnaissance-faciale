import cv2
from ultralytics import YOLO
import os
import json
from flask import Flask, Response
import threading

class_list = ["guns"]

# Initialize the YOLO model
model_path = os.path.join(os.path.dirname(__file__), 'runs/detect/Normal_Compressed/weights/best.pt')

if not os.path.exists(model_path):
    print(f"Erreur : modèle non trouvé à {model_path}")
    exit()

yolo_model = YOLO(model_path)

# Initialize Flask app for mobile integration
app = Flask(__name__)

# Use camera instead of video file
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("Erreur : impossible d'accéder à la caméra")
    exit()

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

count = 0
detection_results = {"detections": [], "weapon_detected": False}

def generate_frames():
    global count, detection_results
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 2 != 0:  # Process every 2nd frame for performance
            continue
        
        results = yolo_model(frame, conf=0.5)
        
        weapon_found = False
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy
            confidences = results[0].boxes.conf
            class_ids = results[0].boxes.cls

            detection_results["detections"] = []
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(class_id)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text_position = (x1, y1 - 10)
                
                if class_id < len(class_list) and class_list[class_id] == "guns":
                    text = f"Weapon detected ({conf:.2f})"
                    weapon_found = True
                else:
                    text = f"Detection ({conf:.2f})"
                
                cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                detection_results["detections"].append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": float(conf),
                    "class": class_list[class_id]
                })
        
        detection_results["weapon_detected"] = weapon_found
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    return detection_results

if __name__ == '__main__':
    print("Serveur démarré sur http://localhost:5000")
    print("Flux vidéo disponible sur http://localhost:5000/video_feed")
    print("Données de détection disponibles sur http://localhost:5000/detections")
    app.run(host='0.0.0.0', port=5000, debug=False)
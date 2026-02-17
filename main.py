import cv2
import numpy as np
import os
import csv
from datetime import datetime
from liveness import LivenessDetector 

# --- 1. SETUP & TRAINING ---
print("[INFO] Training the face recognizer...")

# Create the specific OpenCV Face Recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    names = {} 
    
    for image_path in image_paths:
        if image_path.endswith(".jpg"):
            # Read image in grayscale
            img_numpy = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Extract ID from filename (format: "Name.ID.Count.jpg")
            filename = os.path.split(image_path)[-1]
            parts = filename.split(".")
            
            # Simple error check to make sure filename format is correct
            if len(parts) > 2:
                name = parts[0]
                try:
                    id = int(parts[1])
                except ValueError:
                    continue # Skip files that don't match format
                
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(id)
                    names[id] = name
                
    return face_samples, ids, names

# Train the model
if os.path.exists("dataset"):
    faces, ids, names_map = get_images_and_labels('dataset')
    if len(ids) > 0:
        recognizer.train(faces, np.array(ids))
        print("[SUCCESS] Training complete!")
    else:
        print("[ERROR] No face images found in 'dataset' folder. Run register_face.py!")
        exit()
else:
    print("[ERROR] Dataset folder missing.")
    exit()

# --- 2. MAIN ATTENDANCE LOOP ---
ATTENDANCE_FILE = "attendance_log.csv"
cap = cv2.VideoCapture(0)
liveness_detector = LivenessDetector()

# Create CSV if needed
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Action", "Time", "Date"])

def log_attendance(name, action):
    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, action, now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")])
    print(f"\n[LOG] {action} for {name}")

blink_count = 0
is_real_person = False
current_name = "Unknown"

print("\n--- SYSTEM READY ---")
print("1. Look at camera. BLINK eyes to verify liveness (Red -> Green).")
print("2. Press 'i' to Punch IN.")
print("3. Press 'o' to Punch OUT.")
print("4. Press 'q' to Quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    # Liveness Logic
    # Liveness Logic
    # We only check for blinks if a face is present to avoid false positives
    if len(faces) > 0:
        if liveness_detector.is_blinking(frame):
            blink_count += 1
            # Require a few frames of "closed eyes" to confirm
            if blink_count > 5: 
                is_real_person = True
        else:
            # Reset if eyes are open (optional, or just keep counting)
            pass
        # Require 3 blinks to pass
        if blink_count > 3: 
            is_real_person = True

    # Display Liveness Status
    if is_real_person:
        cv2.putText(frame, "LIVENESS: VERIFIED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "LIVENESS: BLINK EYES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Face Recognition
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Predict Identity
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Confidence check (Lower is better in LBPH)
        if confidence < 70: 
            current_name = names_map.get(id, "Unknown")
            
            # Only show name if Liveness is also verified
            if is_real_person:
                color = (0, 255, 0)
            else:
                color = (0, 165, 255) # Orange if face known but liveness failed
                
            cv2.putText(frame, current_name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            current_name = "Unknown"

    cv2.imshow('Attendance System', frame)
    
    key = cv2.waitKey(10) & 0xff
    if key == ord('i'):
        if is_real_person and current_name != "Unknown":
            log_attendance(current_name, "PUNCH-IN")
            print("[INFO] Attendance marked. Closing system...")
            break  # <--- THIS STOPS THE LOOP AND CLOSES THE WINDOW
    elif key == ord('o'):
        if is_real_person and current_name != "Unknown":
            log_attendance(current_name, "PUNCH-OUT")
            is_real_person = False
            blink_count = 0
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
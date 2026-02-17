import cv2
import os
import numpy as np

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

def register_user():
    name = input("Enter user name: ")
    id = input("Enter a numeric ID (e.g., 1): ")
    
    # Load the standard face detector
    # This comes built-in with the library you just installed!
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    count = 0

    print("\n[INFO] Look at the camera. Capturing face data...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            # Save the captured face image
            cv2.imwrite(f"dataset/{name}.{id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.waitKey(100) # Small delay to capture varied angles

        cv2.imshow("Register Face", frame)
        cv2.putText(frame, f"Captured: {count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if cv2.waitKey(1) == 13 or count == 50: # 13 is Enter key
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SUCCESS] Collection complete for {name}!")

if __name__ == "__main__":
    register_user()
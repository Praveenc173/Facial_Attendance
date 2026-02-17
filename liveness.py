import cv2

class LivenessDetector:
    def __init__(self):
        # Use the standard Eye Detector built into OpenCV
        # This requires NO extra libraries.
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def is_blinking(self, frame):
        """
        Returns True if eyes are NOT detected (assumed blink).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes in the image
        # scaleFactor=1.1, minNeighbors=5 are standard tuning params
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
        
        # LOGIC:
        # If the main system sees a face, but this detector sees 0 eyes,
        # we assume the user is blinking.
        if len(eyes) == 0:
            return True
            
        return False
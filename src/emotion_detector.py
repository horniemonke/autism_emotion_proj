import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings and version issues
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from deepface import DeepFace
    print("DeepFace imported successfully")
except Exception as e:
    print(f"DeepFace import error: {e}")
    DeepFace = None

from src import config as co

class EmotionDetector:
    """
    Wrapper class for DeepFace emotion detection with face detection using Haar Cascade.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion detector.
        
        Parameters:
            model_path (str): Path to the facial expression model weights.
                             If None, uses default DeepFace model.
        """
        self.model_path = model_path
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, frame):
        """
        Detect faces in the frame using Haar Cascade.
        
        Parameters:
            frame (numpy.ndarray): Input frame/image
            
        Returns:
            list: List of face bounding boxes [(x, y, w, h), ...]
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        print(f"Face detection: Found {len(faces)} faces")
        return faces
    
    def analyze_emotion(self, face_roi):
        """
        Analyze emotion for a single face ROI using DeepFace.
        
        Parameters:
            face_roi (numpy.ndarray): Face region of interest
            
        Returns:
            dict: Emotion analysis results with dominant emotion and confidence scores
        """
        if DeepFace is None:
            print("DeepFace not available")
            return None
            
        try:
            # Convert BGR to RGB for DeepFace
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            print(f"Analyzing emotion for face ROI: {face_roi.shape}")
            
            # Use simple DeepFace.analyze like in working project
            result = DeepFace.analyze(rgb_face, actions=['emotion'], enforce_detection=False)
            
            emotion_result = result[0] if isinstance(result, list) else result
            print(f"Emotion analysis result: {emotion_result.get('dominant_emotion', 'unknown')}")
            return emotion_result
            
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            # Return a default result if analysis fails
            return {
                'dominant_emotion': 'unknown',
                'emotion': {
                    'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0,
                    'sad': 0.0, 'surprise': 0.0, 'neutral': 1.0
                }
            }
    
    def predict(self, frame):
        """
        Complete emotion detection pipeline: detect faces and analyze emotions.
        
        Parameters:
            frame (numpy.ndarray): Input frame/image
            
        Returns:
            list: List of dictionaries containing face info and emotion results
        """
        results = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Analyze emotion
            emotion_result = self.analyze_emotion(face_roi)
            
            if emotion_result:
                result = {
                    'bounding_box': (x, y, w, h),
                    'emotion': emotion_result['dominant_emotion'],
                    'emotion_scores': emotion_result['emotion'],
                    'confidence': max(emotion_result['emotion'].values())
                }
                results.append(result)
            else:
                # If DeepFace is not available, still show face detection
                result = {
                    'bounding_box': (x, y, w, h),
                    'emotion': 'face_detected',
                    'emotion_scores': {'face_detected': 1.0},
                    'confidence': 1.0
                }
                results.append(result)
        
        return results
    
    def draw_results(self, frame, results):
        """
        Draw emotion detection results on the frame.
        
        Parameters:
            frame (numpy.ndarray): Input frame
            results (list): List of detection results
            
        Returns:
            numpy.ndarray: Frame with drawn results
        """
        image = frame.copy()
        
        for result in results:
            x, y, w, h = result['bounding_box']
            emotion = result['emotion']
            confidence = result['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw emotion label with confidence
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image

import cv2
import numpy as np

def draw_bbox(image, bbox, score, label=""):
    """
    Draw bounding box and score on the image.

    Parameters:
        image (numpy.ndarray): Input image.
        bbox (tuple): Bounding box (x, y, w, h).
        score (float): Confidence score for the bounding box.
        label (str): Additional label text.

    Returns:
        image: Image with bounding box and score drawn.
    """
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue in BGR
    
    # Draw score and label
    text = f"{label}: {score:.2f}" if label else f"{score:.2f}"
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Blue in BGR
    return image

def format_emotion_result(results):
    """
    Format emotion detection results for display.
    
    Parameters:
        results (list): List of emotion detection results
        
    Returns:
        str: Formatted result string
    """
    if not results:
        return "No face detected"
    
    if len(results) == 1:
        result = results[0]
        emotion = result['emotion']
        confidence = result['confidence']
        
        # Handle special case for face_detected
        if emotion == 'face_detected':
            return f"Face detected\n(Confidence: {confidence:.2f})"
        else:
            return f"{emotion.title()}\n(Confidence: {confidence:.2f})"
    else:
        # Multiple faces - show count and main emotions
        emotions = [r['emotion'] for r in results]
        return f"{len(results)} faces detected\nMain: {emotions[0]}"

def resize_frame(frame, max_width=800, max_height=600):
    """
    Resize frame while maintaining aspect ratio.
    
    Parameters:
        frame (numpy.ndarray): Input frame
        max_width (int): Maximum width
        max_height (int): Maximum height
        
    Returns:
        numpy.ndarray: Resized frame
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    return frame

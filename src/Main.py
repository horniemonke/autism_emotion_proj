# coding=utf-8
import os
import json, time
import threading
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor
from qt_thread_updater import get_updater
from src import config as co
from src.emotion_detector import EmotionDetector
from src.utils import draw_bbox, format_emotion_result, resize_frame

def text_size(frame):
    """Calculate text size based on frame dimensions."""
    frame_height, frame_width = frame.shape[:2]

    # Define the text and initial font settings
    text = "Emotion Detection"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate font scale and thickness based on frame size
    font_scale = frame_width / 600  # Adjust this divisor to scale text size
    font_thickness = max(2, int(frame_height / 200))  # Ensure thickness >= 1
    text_color = (255, 0, 0)  # Blue in BGR

    # Calculate text size with the dynamic scale and thickness
    (_, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Define text position in the top-left corner with padding
    padding = int(0.02 * frame_height)  # 2% of the frame height as padding
    text_x = padding
    text_y = text_height + padding  # Position Y to account for text height

    return (text_x, text_y), font, font_scale, text_color, font_thickness

class Main:
    def __init__(self, MainGUI):
        self.MainGUI = MainGUI
        self.camera = None
        self.ret = False
        self.start_camera = True
        
        # FPS control
        self.target_fps = 10  # Target FPS for processing
        self.frame_skip = 0
        self.frame_count = 0
        
        # Initialize emotion detector with configurable thresholds
        model_path = co.FACIAL_EXPRESSION_MODEL if os.path.exists(co.FACIAL_EXPRESSION_MODEL) else None
        self.emotion_detector = EmotionDetector(
            model_path=model_path,
            min_neighbors=co.FACE_DETECTION_MIN_NEIGHBORS,
            scale_factor=co.FACE_DETECTION_SCALE_FACTOR,
            min_face_size=co.FACE_DETECTION_MIN_SIZE,
            emotion_confidence_threshold=co.EMOTION_CONFIDENCE_THRESHOLD
        )
        
        self.init_text_size()

    def img_cv_2_qt(self, img_cv):
        """
        Convert OpenCV image to Qt image with letterbox padding (gray fill).
        Resizes image to fit widget while maintaining aspect ratio, then centers it
        on a gray background to fill the entire widget area.
        """
        # Get widget dimensions
        widget_w = self.MainGUI.label_Image.width()
        widget_h = self.MainGUI.label_Image.height()
        
        # Convert OpenCV BGR image to QPixmap
        img_height, img_width, channel = img_cv.shape
        bytes_per_line = channel * img_width
        img_qt = QtGui.QImage(img_cv, img_width, img_height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        original_pixmap = QtGui.QPixmap.fromImage(img_qt)
        
        # Calculate aspect ratios
        img_aspect = img_width / img_height if img_height > 0 else 1.0
        widget_aspect = widget_w / widget_h if widget_h > 0 else 1.0
        
        # Calculate new dimensions to fit widget while maintaining aspect ratio
        if img_aspect > widget_aspect:
            # Image is wider - fit to widget width
            new_width = widget_w
            new_height = int(widget_w / img_aspect)
        else:
            # Image is taller - fit to widget height
            new_height = widget_h
            new_width = int(widget_h * img_aspect)
        
        # Resize original image maintaining aspect ratio
        scaled_pixmap = original_pixmap.scaled(
            new_width, 
            new_height, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Create letterbox pixmap with exact widget size and gray fill
        letterbox_pixmap = QtGui.QPixmap(widget_w, widget_h)
        letterbox_pixmap.fill(QColor(128, 128, 128))  # Gray background (RGB: 128, 128, 128)
        
        # Calculate offset to center the scaled image
        offset_x = (widget_w - new_width) // 2
        offset_y = (widget_h - new_height) // 2
        
        # Draw scaled image onto letterbox at center position
        painter = QPainter(letterbox_pixmap)
        painter.drawPixmap(offset_x, offset_y, scaled_pixmap)
        painter.end()
        
        return letterbox_pixmap
    
    def init_devices(self, url_camera):
        """Initialize camera or video capture device."""
        self.camera = cv2.VideoCapture(url_camera) 
        self.ret, frame = self.camera.read() 
        if not self.ret:
            self.start_camera = False
            self.MainGUI.MessageBox_signal.emit("Có lỗi xảy ra ! \n Không tìm thấy camera/video", "error")
        else:
            self.start_camera = True
            (self.text_x, self.text_y), self.font, self.font_scale, self.text_color, self.font_thickness = text_size(frame)
    
    def auto_camera(self):
        """Real-time emotion detection from camera."""
        url_camera = co.CAMERA_DEVICE
        self.init_devices(url_camera)
        
        # Calculate frame skip based on target FPS
        camera_fps = 30  # Assume camera runs at 30 FPS
        self.frame_skip = max(1, camera_fps // self.target_fps)
        print(f"Camera FPS control: Processing every {self.frame_skip} frames (target: {self.target_fps} FPS)")
        
        while self.ret and self.start_camera:
            try:
                ret, frame = self.camera.read()
                self.ret = ret
                if self.ret and self.start_camera:
                    self.frame_count += 1
                    
                    # Skip frames to control FPS
                    if self.frame_count % self.frame_skip != 0:
                        continue
                    
                    # Resize frame for better performance
                    frame = resize_frame(frame)
                    
                    # Detect emotions
                    results = self.emotion_detector.predict(frame)
                    
                    # Draw results
                    image = self.emotion_detector.draw_results(frame, results)
                    
                    # Add title with FPS info
                    title_text = f"Emotion Detection (FPS: {self.target_fps})"
                    cv2.putText(image, title_text, 
                               (self.text_x, self.text_y), 
                               self.font, self.font_scale, 
                               self.text_color, self.font_thickness)
                    
                    # Update UI
                    get_updater().call_latest(self.MainGUI.label_Image.setPixmap, self.img_cv_2_qt(image))
                    
                    # Update result text
                    if results:
                        result_text = format_emotion_result(results)
                        get_updater().call_latest(self.MainGUI.text_result.setText, result_text)
                        get_updater().call_latest(self.MainGUI.text_result.setStyleSheet, "background-color: rgb(0, 255, 0);")
                    else:
                        get_updater().call_latest(self.MainGUI.text_result.setText, "No face detected")
                        get_updater().call_latest(self.MainGUI.text_result.setStyleSheet, "background-color: rgb(255, 255, 0);")
                else:
                    break
            except Exception as e:
                print("Bug: ", e)
        self.close_camera()

    def auto_video(self, path_video):
        """Emotion detection from video file."""
        url_camera = path_video
        self.init_devices(url_camera)
        
        # Get video FPS and calculate frame skip
        video_fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.frame_skip = max(1, int(video_fps // self.target_fps))
        print(f"Video FPS control: Original FPS: {video_fps}, Processing every {self.frame_skip} frames (target: {self.target_fps} FPS)")
        
        while self.ret and self.start_camera:
            try:
                ret, frame = self.camera.read()
                self.ret = ret
                if self.ret and self.start_camera:
                    self.frame_count += 1
                    
                    # Skip frames to control FPS
                    if self.frame_count % self.frame_skip != 0:
                        continue
                    
                    # Resize frame for better performance
                    frame = resize_frame(frame)
                    
                    # Detect emotions
                    results = self.emotion_detector.predict(frame)
                    
                    # Draw results
                    image = self.emotion_detector.draw_results(frame, results)
                    
                    # Add title with FPS info
                    title_text = f"Video Emotion Detection (FPS: {self.target_fps})"
                    cv2.putText(image, title_text, 
                               (self.text_x, self.text_y), 
                               self.font, self.font_scale, 
                               self.text_color, self.font_thickness)
                    
                    # Update UI
                    get_updater().call_latest(self.MainGUI.label_Image.setPixmap, self.img_cv_2_qt(image))
                    
                    # Update result text
                    if results:
                        result_text = format_emotion_result(results)
                        get_updater().call_latest(self.MainGUI.text_result.setText, result_text)
                        get_updater().call_latest(self.MainGUI.text_result.setStyleSheet, "background-color: rgb(0, 255, 0);")
                    else:
                        get_updater().call_latest(self.MainGUI.text_result.setText, "No face detected")
                        get_updater().call_latest(self.MainGUI.text_result.setStyleSheet, "background-color: rgb(255, 255, 0);")
                else:
                    break
            except Exception as e:
                print("Bug: ", e)
        self.close_camera()

    def manual_image(self, image_file):
        """Emotion detection from single image."""
        try:
            frame = cv2.imread(image_file)
            if frame is None:
                self.MainGUI.MessageBox_signal.emit("Không thể đọc file ảnh!", "error")
                return
            
            # Resize frame for better performance
            frame = resize_frame(frame)
            
            # Detect emotions
            results = self.emotion_detector.predict(frame)
            
            # Draw results
            image = self.emotion_detector.draw_results(frame, results)
            
            # Add title
            cv2.putText(image, "Image Emotion Detection", 
                       (self.text_x, self.text_y), 
                       self.font, self.font_scale, 
                       self.text_color, self.font_thickness)
            
            # Update UI
            get_updater().call_latest(self.MainGUI.label_Image.setPixmap, self.img_cv_2_qt(image))
            
            # Update result text
            if results:
                result_text = format_emotion_result(results)
                get_updater().call_latest(self.MainGUI.text_result.setText, result_text)
                get_updater().call_latest(self.MainGUI.text_result.setStyleSheet, "background-color: rgb(0, 255, 0);")
            else:
                get_updater().call_latest(self.MainGUI.text_result.setText, "No face detected")
                get_updater().call_latest(self.MainGUI.text_result.setStyleSheet, "background-color: rgb(255, 0, 0);")
                
        except Exception as e:
            self.MainGUI.MessageBox_signal.emit(f"Lỗi xử lý ảnh: {str(e)}", "error")

    def close_camera(self):
        """Close camera and cleanup resources."""
        try:
            self.start_camera = False
            if self.ret:
                self.camera.release()
            self.camera = None
            self.ret = False
            
            time.sleep(1)
            self.MainGUI.label_Image.clear()

        except Exception as e:
                print("Bug: ", e)

    def init_text_size(self):
        """Initialize default text size parameters."""
        self.text_x = 20
        self.text_y = 20
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.text_color = (255, 0, 0)  # Blue in BGR
        self.font_thickness = 2
    
    def set_target_fps(self, fps):
        """Set target FPS for processing."""
        self.target_fps = max(1, min(30, fps))  # Limit between 1-30 FPS
        print(f"Target FPS set to: {self.target_fps}")

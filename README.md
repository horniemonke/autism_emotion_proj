# Autism Emotion AI

**An AI-powered application for emotion recognition and education for children with autism**

This project develops an AI application using computer vision technology to recognize emotions in children with autism, supporting emotional education and improving their communication skills. The system not only helps children better understand their own emotions but also provides visual guidance to help children develop social interaction skills more naturally and effectively.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Emotion Detection Capabilities](#emotion-detection-capabilities)
- [Skills & Learning Outcomes](#skills--learning-outcomes)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Screenshots](#screenshots)

---

## Overview

Autism Emotion AI is a desktop application that uses deep learning and computer vision to detect and analyze facial emotions in real-time. The application supports multiple input modes (camera, video files, and images) and provides visual feedback to help children with autism understand and recognize emotions.

**Target Audience:** Students oriented towards Computer Science, Engineering, and Data Science fields.

---

## Features

- **Real-time Emotion Detection**: Live emotion recognition from webcam with FPS control (10 FPS target)
- **Video Processing**: Emotion analysis from video files (MP4, AVI, WMV, MKV, MOV)
- **Image Analysis**: Single image emotion detection support (PNG, JPG, JPEG, BMP, TIFF)
- **Multi-face Detection**: Detects and analyzes emotions for multiple faces simultaneously
- **Visual Feedback**: Real-time visualization with bounding boxes and emotion labels
- **System Monitoring**: Built-in CPU, RAM, and Disk usage monitoring
- **User-friendly GUI**: Intuitive PyQt5 interface designed for accessibility

---

## Emotion Detection Capabilities

The application recognizes **7 emotion states**:

1. **Happy** 😊
2. **Neutral** 😐
3. **Surprise** 😲
4. **Sad** 😢
5. **Angry** 😠
6. **Fear** 😨
7. **Disgust** 🤢

Each emotion is detected with a confidence score, and the dominant emotion is displayed prominently.

---

## Skills & Learning Outcomes

Through this project, students will achieve:

- **Critical Thinking & Problem-Solving**: Analyze real-world requirements, reason for topic selection, research technologies, and connect features into a meaningful application
- **AI Computer Vision Programming**: Develop emotion recognition models, emotion scoring systems, and learn the complete application development process from data preparation to result evaluation through hands-on practice
- **UI/UX Design & Development**: Form initial thinking about designing intuitive application interfaces for children with autism

---

## Technology Stack

- **GUI Framework**: PyQt5, PyQt5-sip, PyQtWebEngine
- **AI/ML**: TensorFlow 2.16.1, DeepFace, tf_keras
- **Computer Vision**: OpenCV 4.5.5.64
- **Image Processing**: Pillow 9.5.0
- **System Monitoring**: psutil 5.9.8
- **Thread Management**: qt-thread-updater 1.1.6
- **Data Processing**: NumPy 1.23.5, Pandas, Matplotlib, Seaborn
- **Utilities**: PyYAML 6.0, imutils, tqdm, requests

---

## Project Structure

```
autism_emotion_proj/
├── MainGUI.py              # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # License file
│
├── src/                   # Core source code
│   ├── __init__.py
│   ├── Main.py           # Main processing logic (camera/video/image)
│   ├── emotion_detector.py  # Emotion detection wrapper (DeepFace + Haar Cascade)
│   ├── config.py         # Configuration paths and settings
│   ├── utils.py          # Utility functions (drawing, formatting, resizing)
│   └── Timer.py          # Multi-threaded timer for periodic tasks
│
├── GUI/                   # GUI resources
│   ├── Main.ui           # Qt Designer UI file
│   ├── ico.ICO           # Application icon
│   ├── icons/            # Button icons
│   └── images/           # Background images
│
└── imgs/                  # Screenshots directory (for documentation)
```

---

## Installation

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Conda** (Anaconda or Miniconda)
- **Webcam** (for real-time detection)

### Step-by-Step Installation

#### 1. Install Conda (if not already installed)

Download and install Anaconda or Miniconda from:
- **Anaconda**: https://www.anaconda.com/products/distribution
- **Miniconda**: https://docs.conda.io/en/latest/miniconda.html

#### 2. Create Conda Environment

Open your terminal (Command Prompt, PowerShell, or Anaconda Prompt) and navigate to the project directory:

```bash
cd path/to/autism_emotion_proj
```

Create a new conda environment with Python 3.9:

```bash
conda create -n autism_emotion python=3.11 -y
```

Activate the environment:

```bash
conda activate autism_emotion
```

#### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note:** The installation may take several minutes, especially for TensorFlow and related packages.

#### 4. Verify Installation

Ensure all packages are installed correctly:

```bash
python -c "import cv2, tensorflow, PyQt5; print('All packages installed successfully!')"
```

---

## Usage

### Running the Application

1. **Activate the conda environment** (if not already activated):
   ```bash
   conda activate autism_emotion
   ```

2. **Navigate to the project directory**:
   ```bash
   cd path/to/autism_emotion_proj
   ```

3. **Run the application**:
   ```bash
   python MainGUI.py
   ```

### Using the Application

#### Camera Mode (Real-time Detection)
1. Click the **Camera** button to start real-time emotion detection from your webcam
2. The application will display live video with emotion labels and bounding boxes
3. Results are shown in real-time in the text area

#### Video Mode
1. Click the **Video** button
2. Select a video file (MP4, AVI, WMV, MKV, MOV)
3. The application will process the video frame by frame and display results

#### Image Mode
1. Click the **Image** button
2. Select an image file (PNG, JPG, JPEG, BMP, TIFF)
3. The application will analyze the image and display emotion detection results

#### Stop Processing
- Click the **Stop** button to stop the current operation and return to the main interface

---

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│                  Application Startup                     │
│              (MainGUI.py - Entry Point)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Initialize Main Class                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  • Load EmotionDetector (DeepFace + Haar Cascade) │  │
│  │  • Initialize Camera/Video/Image handlers         │  │
│  │  • Setup FPS control (target: 10 FPS)             │  │
│  │  • Initialize text rendering parameters            │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            User Selects Input Mode                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Camera     │  │    Video     │  │    Image     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │           │
│         ▼                 ▼                  ▼           │
│  auto_camera()    auto_video(path)    manual_image(path) │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Frame Processing Loop                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  1. Capture/Read frame                             │  │
│  │  2. Resize frame (max 800x600) for performance     │  │
│  │  3. Frame skip control (to maintain 10 FPS)       │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Emotion Detection Pipeline                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  EmotionDetector.predict(frame)                    │  │
│  │                                                    │  │
│  │  1. Face Detection (Haar Cascade)                 │  │
│  │     └─> Returns: [(x, y, w, h), ...]              │  │
│  │                                                    │  │
│  │  2. Emotion Analysis (DeepFace)                     │  │
│  │     └─> For each face ROI:                         │  │
│  │         • Convert BGR to RGB                       │  │
│  │         • DeepFace.analyze(actions=['emotion'])    │  │
│  │         • Extract dominant emotion & scores        │  │
│  │                                                    │  │
│  │  3. Result Formatting                              │  │
│  │     └─> Returns: [{                                │  │
│  │           'bounding_box': (x, y, w, h),            │  │
│  │           'emotion': 'happy',                      │  │
│  │           'emotion_scores': {...},                 │  │
│  │           'confidence': 0.85                       │  │
│  │         }, ...]                                    │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Visualization & Display                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  1. Draw Results                                   │  │
│  │     • Draw bounding boxes (green)                  │  │
│  │     • Add emotion labels with confidence           │  │
│  │     • Add title text                               │  │
│  │                                                    │  │
│  │  2. Update UI (Thread-safe)                       │  │
│  │     • Convert OpenCV image to Qt Pixmap           │  │
│  │     • Update label_Image widget                    │  │
│  │     • Update text_result widget                    │  │
│  │     • Update system monitors (CPU/RAM/Disk)        │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │ Continue Loop │
              │  (until stop) │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   Cleanup    │
              │ Close camera │
              │  Release res. │
              └──────────────┘
```

---

## Screenshots

### Main Application Interface

The following screenshot demonstrates the main user interface of the Autism Emotion AI application:

![Main Application Interface](imgs/demo_ui.PNG)

**Features shown:**
- **Image Mode**: Application processing a static image with the "Image" button highlighted in blue
- **Emotion Detection Result**: Successfully detected "happy" emotion with 99.80% confidence
- **Visual Feedback**: Blue bounding box and emotion label displayed on the detected face
- **Aspect Ratio Preservation**: Image letterboxed with gray padding on left and right sides to maintain aspect ratio
- **Result Panel**: Green result bar showing "Happy (Confidence: 99.80)" in the RESULT section
- **Control Panel**: SELECTION buttons for Camera, Video, Image, and Stop controls
- **System Monitoring**: Built-in performance monitoring displayed in the MONITOR section

This screenshot showcases the application's ability to accurately detect emotions in static images while maintaining proper image aspect ratios and providing clear visual feedback.

---

## Troubleshooting

### Common Issues

1. **Camera not detected**
   - Ensure your webcam is connected and not being used by another application
   - Check camera permissions in system settings

2. **TensorFlow/DeepFace import errors**
   - Reinstall TensorFlow: `pip install --upgrade tensorflow==2.16.1`
   - Reinstall DeepFace: `pip install --upgrade deepface`

3. **PyQt5 errors**
   - Ensure all PyQt5 packages are installed: `pip install PyQt5 PyQt5-sip PyQtWebEngine`

4. **Performance issues**
   - The application automatically adjusts FPS to 10 FPS for optimal performance
   - If still slow, try closing other applications to free up system resources

5. **Memory errors**
   - Reduce video resolution or use shorter video clips
   - Close and restart the application periodically

---

## License

See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **DeepFace** - Emotion recognition model
- **OpenCV** - Computer vision library
- **PyQt5** - GUI framework

---

## Contact & Support

For questions, issues, or contributions, please refer to the project repository or contact the development team.

---

**Note**: This application is designed as an educational tool to support emotion recognition and learning for children with autism. It should be used as part of a comprehensive therapeutic and educational program under professional guidance.

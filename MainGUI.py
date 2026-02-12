import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, uic, QtCore, QtGui, QtWidgets
import sys
import traceback
import multiprocessing
import time
import psutil
from threading import Thread
import json
from qt_thread_updater import get_updater
from src.Main import Main
from src import config as co, Timer

class MainGUI(QtWidgets.QMainWindow):
    MessageBox_signal = QtCore.pyqtSignal(str, str)
    
    def __init__(self):
        super(MainGUI, self).__init__()
        self.ui = uic.loadUi(co.MAIN_GUI, self)
        
        # Connect button signals
        self.pushButton_Camera.clicked.connect(self.open_camera)
        self.pushButton_Video.clicked.connect(self.open_video)
        self.pushButton_Image.clicked.connect(self.manual)
        self.pushButton_Stop.clicked.connect(self.stop)
        self.MessageBox_signal.connect(self.MessageBox_slot)
        
    def start(self):
        """Initialize and start the application."""
        try: 
            self.show()
            self.Main = Main(self.ui)
            Timer.Timer(function=self.monitor_pc_performance, name="pc_performance", forever=True, interval=2, type="repeat").start()
        except Exception as e:
            self.MessageBox_signal.emit(str(e), "error")
            sys.exit(1)
    
    def open_camera(self):
        """Start real-time emotion detection from camera."""
        try:
            self.update_window("start", name="auto_camera")
            Timer.Timer(function=self.Main.auto_camera, name="main_auto").start()
        except Exception as e:
            self.MessageBox_signal.emit(str(e), "error")
            self.MessageBox_signal.emit("Có lỗi xảy ra !", "error")
            self.stop()
    
    def open_video(self):
        """Start emotion detection from video file."""
        try:
            self.update_window("start", name="auto_video")
            options = QtWidgets.QFileDialog.Options()
            video_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 
                "Chọn file video", 
                "", 
                "Video (*.mp4 *.avi *.wmv *.mkv *.mov)", 
                options=options
            )
            if video_file:
                Timer.Timer(function=self.Main.auto_video, name="auto_video", args=[video_file]).start()

        except Exception as e:
            self.MessageBox_signal.emit(str(e), "error")
            self.MessageBox_signal.emit("Có lỗi xảy ra !", "error")
            self.stop()

    def manual(self):
        """Process single image for emotion detection."""
        try:
            self.update_window("start", name="manual")
            options = QtWidgets.QFileDialog.Options()
            img_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 
                "Chọn file ảnh", 
                "", 
                "Images (*.png *.jpg *.jpeg *.bmp *.tiff)", 
                options=options
            )
            if img_file:
                # Normalize path separators
                img_file = img_file.replace(os.sep, os.altsep)
                temp = img_file.split('/')
                img_file = "\\".join(temp)
                self.Main.manual_image(img_file)

        except Exception as e:
            self.MessageBox_signal.emit(str(e), "error")
            self.MessageBox_signal.emit("Có lỗi xảy ra !", "error")
    
    def stop(self):
        """Stop all running processes."""
        self.update_window("stop")
        time.sleep(1)
        self.Main.close_camera()
    
    def update_window(self, typ, name="auto_camera"):
        """Update UI state based on current operation."""
        if typ == "start":
            if name == "manual":
                self.pushButton_Image.setStyleSheet("background-color: rgb(0, 204, 255);")
                for item in (self.pushButton_Camera, self.pushButton_Video, self.pushButton_Image):
                    item.setEnabled(False)
                self.pushButton_Stop.setEnabled(True)
            else:
                self.pushButton_Stop.setEnabled(True)
                if name == "auto_camera":
                    self.pushButton_Camera.setStyleSheet("background-color: rgb(0, 204, 255);")
                    for item in (self.pushButton_Camera, self.pushButton_Video):
                        item.setEnabled(False)
                elif name == "auto_video":
                    self.pushButton_Video.setStyleSheet("background-color: rgb(0, 204, 255);")
                    for item in (self.pushButton_Camera, self.pushButton_Video):
                        item.setEnabled(False)

        elif typ == "stop":
            for item in [self.pushButton_Stop]:
                item.setEnabled(False)
            for item in (self.pushButton_Camera, self.pushButton_Video, self.pushButton_Image):
                item.setEnabled(True)
                item.setStyleSheet("")
   
    def closeEvent(self, event):
        """Handle application close event."""
        reply = QtWidgets.QMessageBox.question(
            self, 
            "Thông báo", 
            "Bạn có chắc chắn muốn thoát không ?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
            QtWidgets.QMessageBox.Yes
        )
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def MessageBox_slot(self, txt, style):
        """Display message box based on style."""
        if style == "error":
            QtWidgets.QMessageBox.critical(self, "Lỗi", txt)
        elif style == "warning":
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", txt)
        else:
            QtWidgets.QMessageBox.information(self, "Thông báo", txt)

    def monitor_pc_performance(self):
        """Monitor system performance and update UI."""
        try:
            cpu_percent = sum(psutil.cpu_percent(percpu=True))/psutil.cpu_count()
            mem_stats = psutil.virtual_memory()
            disk_stats = psutil.disk_usage("/")   
            get_updater().call_latest(self.progressBar_CPU.setValue, int(cpu_percent)) 
            get_updater().call_latest(self.progressBar_RAM.setValue, int(mem_stats.percent))
            get_updater().call_latest(self.progressBar_DISK.setValue, int(disk_stats.percent))
            get_updater().call_latest(self.ram.setText, f"{round(mem_stats.used/1000000000, 1)}/{round(mem_stats.total/1000000000, 1)}")
            get_updater().call_latest(self.disk.setText, f"{round(disk_stats.used/1000000000, 1)}/{round(disk_stats.total/1000000000)}")
        except Exception as e:
            pass

PROCNAME = "python.exe"
def kill_orphan_process():
    """Kill orphaned Python processes."""
    curr_pid = os.getpid()
    for proc in psutil.process_iter():
        if proc.name() == PROCNAME and proc.pid != curr_pid:
            proc.kill()

if __name__ == "__main__":
    # kill_orphan_process()
    app = QApplication(sys.argv)
    main = MainGUI()
    main.start()
    sys.exit(app.exec_())

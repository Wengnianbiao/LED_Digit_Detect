import cv2
from detect_code.main import img_detect


def digitaldetect(path):
    cap = cv2.VideoCapture(path)

# Individual project
# P15/139992/2020 GITARI JOAN WANJIRU
# COMPUTER VISION: OBJECT DETECTION
# OpenCV

# main application to run

from detector import *
import os

def main():
    # videoPath = "images/pexels-elena-gram-9357246.mp4"
    videoPath = 0

    configPath = os.path.join("models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("models/frozen_inference_graph.pb")
    classesPath = os.path.join("models/coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == "__main__":
    main()
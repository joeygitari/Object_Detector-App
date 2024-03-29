# Individual project
# P15/139992/2020 GITARI JOAN WANJIRU
# COMPUTER VISION: OBJECT DETECTION
# OpenCV

# detects objects but doesn't give confidence 
 
import cv2 #type: ignore
import cvlib as cv #type: ignore
from cvlib.object_detection import draw_bbox #type: ignore
from gtts import gTTS #type: ignore
from playsound import playsound #type: ignore

# convert speech to text
def speech(text):
    print(text)
    language = "en"
    output = gTTS(text=text, lang=language, slow=False)
    output.save("sounds/output.mp3")
    playsound("sounds/output.mp3")


video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()
    frame = cv2.flip(frame,1)

    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, label, conf)

    cv2.imshow("Object Detection by Joanne", output_image)

    for item in label:
        if item in labels:
            pass
        else:
            labels.append(item)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

i = 0
new_sentence = []
for label in labels:
    if i == 0:
        new_sentence.append(f"I found a {label}, and, ")
    else:
        new_sentence.append(f"a {label},")

    i += 1

speech(" ".join(new_sentence))

video.release()
cv2.destroyAllWindows()    
modelpath = "C:/Users/Lori/Documents/Courses/master/Year2/Sem1/CV/yolo.h5" #local path

from imageai import Detection
yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel()

import cv2
import pandas as pd

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

column_names = ["ID", "Frame", "Object category", "Rectangle Coordinates"]
df = pd.DataFrame(columns=column_names)

ID = 1
frame = 1

while True:
    # read frames
    ret, img = cam.read()
    # predict yolo
    img, preds = yolo.detectObjectsFromImage(input_image=img,
                                          input_type="array",
                                          output_type="array",
                                          minimum_percentage_probability=70,
                                          display_percentage_probability=False,
                                          display_object_name=True)
    # display predictions
    cv2.imshow("", img)
    for p in preds:
        df.loc[ID] = [ID, frame, p["name"], p["box_points"]]
        ID = ID + 1
    frame = frame + 1

    # press q or Esc to quit
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
        break

# close camera
cam.release()
cv2.destroyAllWindows()
df.to_csv("info.csv", index=False)

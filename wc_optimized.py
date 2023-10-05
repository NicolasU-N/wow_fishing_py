import uuid  # Unique identifier
import os
import time
import pyautogui
import cv2
import numpy as np
import time
import torch
from matplotlib import pyplot as plt
from WindowCapture import WindowCapture

IMAGES_PATH = os.path.join("data", "images")  # /data/images
# labels = ['left_obs', 'right_obs', 'timberman']
number_imgs = 250

wc = WindowCapture("Discord", 640, 640)

# Loop over the frames
for img_num in range(number_imgs):
    print("Collecting image number {}".format(img_num))
    # screen = pyautogui.screenshot()
    # screen_array = np.array(screen)
    # # Recorte de 640x640 pixeles de la parte centro superior
    # cropped_region = screen_array[0:640, 640:1280, :]
    # imgname = os.path.join(IMAGES_PATH, str(uuid.uuid1()) + ".jpg")
    # corrected_colors = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(imgname, corrected_colors)
    # results = model(corrected_colors)
    # cv2.imshow('YOLO', np.squeeze(results.render()))

    screenshot = wc.get_screenshot()
    cv2.imshow("YOLO", screenshot)

    # 0.3 second delay between captures
    time.sleep(0.3)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

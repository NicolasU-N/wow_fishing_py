import argparse
import cv2
import numpy as np
import pyautogui
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
import time
import pydirectinput  # Paso 1
from timeit import default_timer as timer

# Cargar las clases y colores
CLASSES = yaml_load(check_yaml("dataset.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Variable global para el conteo
hook_ok_count = 0
last_detection_time = timer()  # Paso 1: Inicialización
last_key2_press_time = timer()  # Paso 1: Inicialización para tecla '2'


def press_keys():
    # pydirectinput.press("h")
    # pyautogui.typewrite("h")
    time.sleep(1)  # Espera de !50ms
    pyautogui.typewrite("1")
    # pydirectinput.press("1")


def check_reset_detection():
    global last_detected_time
    current_time = time.time()
    print(f"time: {current_time - last_detected_time}")
    if (
        last_detected_time is None or (current_time - last_detected_time) > 18
    ):  # Han pasado 18 segundos
        # pydirectinput.press("1")
        pyautogui.typewrite("1")
        time.sleep(0.5)  # Espera de 50ms


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), 3, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 3, 2)


def process_image(onnx_model, input_image):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    original_image = input_image
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1 / 255, size=(640, 640), swapRB=True
    )
    model.setInput(blob)
    outputs = model.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
            classes_scores
        )
        if maxScore >= 0.55:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )
    # print("detections -> ", detections)
    # Verificar detecciones
    global hook_ok_count, last_detection_time  # Usamos las variables globales
    for detection in detections:
        # or detection["class_id"] == 0
        if detection["class_name"] == "hook_ok":
            hook_ok_count += 1
            last_detection_time = timer()  # Paso 3: Actualización del tiempo

            # Calcula el centro del recuadro
            x_center = round(
                (detection["box"][0] + (detection["box"][2] / 2)) * detection["scale"]
            )
            y_center = round(
                (detection["box"][1] + (detection["box"][3] / 2)) * detection["scale"]
            )

            # Mueve el mouse al centro del recuadro y hace un click derecho
            pyautogui.moveTo(x_center + 640, y_center)
            pyautogui.rightClick()

            press_keys()
            print(f"hook_ok detected {hook_ok_count} times.")
            pyautogui.moveTo(20, 20)

    return original_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="./training_results/wow_bot_yolov8n_v42/weights/best.onnx",
        help="Input your onnx model.",
    )
    args = parser.parse_args()

    # pydirectinput.press("2")
    pyautogui.moveTo(20, 20)
    pyautogui.leftClick()

    pyautogui.typewrite("2")
    time.sleep(4)  # Espera de 50ms
    pyautogui.typewrite("1")

    while True:
        # Take a screenshot
        current_time = timer()  # Obtener tiempo actual
        # Verificación de 18 segundos sin detección
        if current_time - last_detection_time > 18:
            # pydirectinput.press("1")
            pyautogui.typewrite("1")
            time.sleep(0.4)  # Espera de 50ms
            last_detection_time = current_time
            print("18 seconds without detection. Pressed '1'.")

        # Paso 2: Verificación de 10.09 minutos para presionar la tecla '2'
        if current_time - last_key2_press_time > (
            10.09 * 60
        ):  # 10.5 minutos convertidos a segundos
            # pydirectinput.press("2")
            pyautogui.typewrite("2")
            time.sleep(4)  # Espera de 50ms
            pyautogui.typewrite("1")
            last_key2_press_time = current_time
            print("10.09 minutes passed. Pressed '2'.")

        screen = pyautogui.screenshot()
        screen_array = np.array(screen)
        cropped_region = screen_array[0:640, 640:1280, :]
        corrected_colors = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR)

        # Process and get results
        processed_img = process_image(args.model, corrected_colors)
        # cv2.imshow("YOLO", processed_img)

        time.sleep(0.25)

        # Wait 0.3 seconds (300 ms) and check for exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

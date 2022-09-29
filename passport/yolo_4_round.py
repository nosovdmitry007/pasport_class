import cv2
import numpy as np
import time
import math
from automaticfilter import auto_rotait
from yolo_4 import yolo_4
from oblasty_yolo_4 import oblasty_yolo_4
import pandas as pd
def zero( n ):
    return n * (n > 0)

def rotate_image(mat, angle):
    """
   Функция для поворота изображений (серийный номер)
    """

    height, width = mat.shape[:2] # форма изображения имеет 3 измерения
    image_center = (width/2, height/2) # getRotationMatrix2D нужны координаты в обратном порядке (ширина, высота) по сравнению с формой

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # вращение вычисляет cos и sin, принимая их абсолютные значения.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # найдите новые границы ширины и высоты
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    #вычтите старый центр изображения (возвращая изображение в исходное состояние) и добавьте новые координаты центра изображения.
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # поверните изображение с новыми границами и преобразованной матрицей поворота
    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

def yolo_4_round(put):
    start_time = time.time()
    # Load Yolo
    net_round = cv2.dnn.readNet("./yolo_round/yolov4-obj_last_round.weights", "./yolo_round/yolov4-obj_round.cfg")
    with open("./yolo_round/round.names","r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net_round.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net_round.getUnconnectedOutLayers()]
    # Loading image
    img = cv2.imread(put)
    height, width, channels = img.shape

    # Detecting objects#
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_round.setInput(blob)
    outs = net_round.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    d = []
    l = []
    print(indexes)
    for i in indexes:
        box = boxes[i]
        d.append(classes[class_ids[i]])
        d.append(box)
        d.append(confidences[i])
        flattenlist = lambda d:[item for element in d for item in flattenlist(element)] if type(d) is list else [d]
        l.append(flattenlist(d))
    print(l)
    cat = l[0][0]
    y = int(l[0][2])
    x = int(l[0][1])
    h = int(l[0][4])
    w = int(l[0][3])
    crop= img[zero(y - math.ceil(h * 0.1)):y + math.ceil(h * 1.1), zero(x - math.ceil(w * 0.1)):x + math.ceil(w * 1.1)]
    crop = rotate_image(crop,int(cat))
    cv2.imwrite('crop.jpg', crop)
    auto_rotait(crop)
    print("--- %s seconds yolo_4---" % (time.time() - start_time))
    # oblasty_yolo_4(put,img,z)


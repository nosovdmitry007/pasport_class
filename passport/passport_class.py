import cv2
import numpy as np
import easyocr
import math
class passport:
    def __init__(self):
        self.reader = easyocr.Reader(['ru'],
                        model_storage_directory='EasyOCR/model',
                        user_network_directory='EasyOCR/user_network',
                        recog_network='custom_example',
                        gpu=False) # распознание с дообучением
        self.net = cv2.dnn.readNet("yolo/yolov4-obj_last.weights", "yolo/yolov4-obj.cfg")
        with open("yolo/passport.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.net_round = cv2.dnn.readNet("yolo_round/yolov4-obj_last_round.weights", "yolo_round/yolov4-obj_round.cfg")
        with open("yolo_round/round.names","r") as f:
            self.classes_round = [line.strip() for line in f.readlines()]

    def zero(self,n):
        return n * (n > 0)

    def rotate_image(self,mat, angle):
        """
       Функция для поворота изображений
        """
        height, width = mat.shape[:2]
        image_center = (width / 2,
                        height / 2)
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]
        ser_nom = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return ser_nom

    def yolo_4_round(self,put):
        layer_names = self.net_round.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net_round.getUnconnectedOutLayers()]
        img = cv2.imread(put)
        height, width = img.shape[:2]
        # Детекция изображения
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net_round.setInput(blob)
        outs = self.net_round.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        d = []
        l = []
        for i in indexes:
            box = boxes[i]
            d.append(self.classes_round[class_ids[i]])
            d.append(box)
            d.append(confidences[i])
            flattenlist = lambda d:[item for element in d for item in flattenlist(element)] if type(d) is list else [d]
            l.append(flattenlist(d))
        cat = l[0][0]
        y = int(l[0][2])
        x = int(l[0][1])
        h = int(l[0][4])
        w = int(l[0][3])
        crop= img[self.zero(y - math.ceil(h * 0.4)):y + math.ceil(h * 1.4), self.zero(x - math.ceil(w * 0.4)):x + math.ceil(w * 1.4)]
        crop = self.rotate_image(crop,int(cat))
        return crop

    def reorder(self,myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew

    def biggestContour(self,contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 5000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest,max_area

    # @profile()
    def auto_rotait(self, img):
        heightImg, widthImg = img.shape[:2]
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgThreshold = cv2.Canny(imgBlur, 30, 30)
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # НАХОДИМ ВСЕ КОНТУРЫ
        biggest, maxArea = self.biggestContour(contours)  # НАЙДИМ САМЫЙ БОЛЬШОЙ КОНТУР

        if biggest.size != 0:
            biggest = self.reorder(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32(
                [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        else:
            imgWarpColored = img
        return imgWarpColored

    def yolo_4(self, img):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        height, width = img.shape[:2]
        # детекция изображения
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        d = []
        z = []
        for i in indexes:
            box = boxes[i]
            d.append(self.classes[class_ids[i]])
            d.append(box)
            d.append(confidences[i])
            flattenlist = lambda d: [item for element in d for item in flattenlist(element)] if type(d) is list else [d]
            z.append(flattenlist(d))
            d = []
        return img, z

    def oblasty_yolo_4(self, image, box):
        oblasty = {}
        iss = 0
        plac = 0
        spissok = sorted(box, reverse=False, key=lambda x: x[2])
        for l in spissok:
            cat = l[0]
            y = int(l[2])
            x = int(l[1])
            h = int(l[4])
            w = int(l[3])
            ob = ''
            if ('signature' in cat) or ('photograph' in cat):
                pass
            else:
                if 'issued_by_whom' in cat:
                    ob = cat + '_' + str(iss)
                    iss += 1
                elif 'place_of_birth' in cat:
                    ob = cat + '_' + str(plac)
                    plac += 1
                elif 'series' not in cat:
                    ob = cat
                if ob:
                    oblasty[ob] = image[self.zero(y - math.ceil(h * 0.07)):y + math.ceil(h * 1.3),
                                  self.zero(x - math.ceil(w * 0.1)):x + math.ceil(w * 1.1)]
                if 'series' in cat:
                    ob = cat
                    cropped = image[self.zero(y - math.ceil(h * 0.1)):y + math.ceil(h * 1.1),
                              self.zero(x - math.ceil(w * 0.03)):x + math.ceil(w * 1.03)]
                    oblasty[ob] = self.rotate_image(cropped, 90)
        return oblasty

    def recognition_slovar(self, oblasty):
        data = {}
        data['pasport'] = []
        d = {}
        issued_by_whom = ''
        series_and_number = ''
        place_of_birth = ''
        ver = 0
        for i, v in oblasty.items():
            image = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
            # Для каждого класса устанавливаем свои ограничения на распознания классов
            if 'date' in i or 'code' in i or 'series' in i:
                result = self.reader.readtext(image, allowlist='0123456789-. ')
            elif 'surname' in i or 'name' in i or 'patronymic' in i:
                result = self.reader.readtext(image,
                                         allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-')
            elif 'gender' in i:
                result = self.reader.readtext(image, allowlist='.МУЖЕНмужен')
            else:
                result = self.reader.readtext(image,
                                        allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-"№ .1234567890',
                                        min_size=25)
            pole = ''
            for k in range(len(result)):
                if result[k][2] * 100 >= 35:
                    if str(result[k][1]).isnumeric():
                        if result[k][2] * 100 >= 70:
                            pole = pole + ' ' + str(result[k][1])
                    else:
                            pole = pole + ' ' + str(result[k][1])
            if pole:
                pole = pole.strip()
                if 'issued_by_whom' in i:
                    issued_by_whom = issued_by_whom + pole + ' '
                if 'place_of_birth' in i:
                    place_of_birth = place_of_birth + pole + ' '
                if 'series_and_number' in i:
                    # print(pole)
                    if len(pole) >= 10:
                        if ver < result[k][2]:
                            series_and_number = pole
                if 'issued_by_whom' in i or 'place_of_birth' in i or 'series_and_number' in i:
                    pass
                elif 'date' in i:
                    pole = pole.replace('.', '').replace(' ', '').replace('-', '')
                    pole = pole[:2] + '.' + pole[2:4] + '.' + pole[4:8]
                    d[i.split('.', 1)[0]] = pole.upper().strip()
                elif 'cod' in i:
                    pole = pole.replace(' . ', '').replace(' ', '').replace('-', '')
                    pole = pole[:3] + '-' + pole[3:6]
                    d[i.split('.', 1)[0]] = pole.upper().strip()
                elif 'gender' in i:
                    if 'Е' in pole.upper() or 'Н' in pole.upper():
                        pole = 'ЖЕН.'
                    elif 'У' in pole.upper() or 'М' in pole.upper():
                        pole = 'МУЖ.'
                    d[i.split('.', 1)[0]] = pole.upper().strip()
                else:
                    d[i.split('.', 1)[0]] = pole.replace('  ', ' ').upper().strip()

        place_of_birth = place_of_birth.upper()
        issued_by_whom = issued_by_whom.upper()
        if place_of_birth[:2] == 'C ':
            place_of_birth = place_of_birth.replace('С ', ' С. ')
        if issued_by_whom[:2] == 'C ':
            issued_by_whom = issued_by_whom.replace('С ', ' С. ')
        place_of_birth = place_of_birth.replace('ГОР ', 'ГОР. ')\
                                        .replace(' С ', ' С. ')\
                                        .replace(' Г ', ' Г. ')\
                                        .replace('ОБЛ ', 'ОБЛ. ')\
                                        .replace('ПОС ', 'ПОС. ')\
                                        .replace('ДЕР ', 'ДЕР. ')\
                                        .replace(' . ', '. ')\
                                        .replace(' .', '.')\
                                        .replace('  ', ' ')\
                                        .replace('..', '.')\
                                        .replace('.', '. ')\
                                        .replace('  ', ' ')
        issued_by_whom = issued_by_whom.replace('ГОР ', 'ГОР. ')\
                                        .replace(' С ', ' С. ')\
                                        .replace(' Г ', ' Г. ')\
                                        .replace('ОБЛ ', 'ОБЛ. ')\
                                        .replace('ПОС ', 'ПОС. ')\
                                        .replace('ДЕР ', 'ДЕР. ')\
                                        .replace(' . ', '. ')\
                                        .replace(' .', '.')\
                                        .replace('  ', ' ')\
                                        .replace('..', '.')\
                                        .replace('.', '. ')\
                                        .replace('  ', ' ')
        if series_and_number:
            series_and_number = series_and_number.replace(' ', '')
            if len(series_and_number) == 10:
                series_and_number = series_and_number[:2] + ' ' + series_and_number[2:4] + ' ' + series_and_number[4:10]
            else:
                series_and_number = 'поле распознано не полностью' + series_and_number
        else:
            series_and_number = 'поле не распознано'
        d['issued_by_whom'] = issued_by_whom.strip()
        d['place_of_birth'] = place_of_birth.strip()
        d['series_and_number'] = series_and_number.strip()
        data['pasport'].append(d)
        return data['pasport']




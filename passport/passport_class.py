import cv2
import numpy as np
import easyocr
import json
import math
class passport:
    def __init__(self):
        self.reader = easyocr.Reader(['ru'], recog_network='custom_example', gpu=False)  # распознание с дообучением
        self.net = cv2.dnn.readNet(r"C:\Users\nosov\PycharmProjects\pasport_class\passport\yolov4-obj_last.weights",
                              r"C:\Users\nosov\PycharmProjects\pasport_class\passport\yolov4-obj.cfg")
        with open(r"C:\Users\nosov\PycharmProjects\pasport_class\passport\passport.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
    def reorder(self,myPoints):

        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)

        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]

        return self.myPointsNew

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
        return self.biggest, self.max_area

    # @profile()
    def auto_rotait(self, photo):
        ########################################################################

        heightImg = 640
        widthImg = 455
        ########################################################################

        ph = photo.split('/')[-1]
        pathImage = photo

        img = cv2.imread(pathImage)
        img = cv2.resize(img, (widthImg, heightImg))  # ИЗМЕНЕНИЕ РАЗМЕРА ИЗОБРАЖЕНИЯ

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ПРЕОБРАЗОВАНИЕ ИЗОБРАЖЕНИЯ В ОТТЕНКИ СЕРОГО
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ДОБАВИТЬ РАЗМЫТИЕ ПО ГАУССУ
        imgThreshold = cv2.Canny(imgBlur, 20, 20)  # thres[0],thres[1]) # ПРИМЕНИТЕ ХИТРОЕ РАЗМЫТИЕ
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # ПРИМЕНИТЕ РАСШИРЕНИЕ
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # НАНЕСИТЕ ЭРОЗИЮ

        ## ## НАЙТИ ВСЕ КОНТУРЫ
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # НАЙТИ ВСЕ КОНТУРЫ

        # # НАЙДИТЕ САМЫЙ БОЛЬШОЙ КОНТУР
        biggest, maxArea = self.biggestContour(contours)  # НАЙДИТЕ САМЫЙ БОЛЬШОЙ КОНТУР
        if biggest.size != 0:
            biggest = self.reorder(biggest)
            pts1 = np.float32(biggest)  # ПОДГОТОВЬТЕ ТОЧКИ ДЛЯ ДЕФОРМАЦИИ
            pts2 = np.float32(
                [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # ПОДГОТОВЬТЕ ТОЧКИ ДЛЯ ДЕФОРМАЦИИ
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        else:
            print(
                'некоректная фотография, необходимо сфотографировать паспорт на однородном фоне\n фотография не обрезалась изменен размер')
            imgWarpColored = img
        # cv2.imwrite('oblosty/' + ph, imgWarpColored)
        return self.imgWarpColored

    def yolo_4(self, put):
        # Load Yolo

        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        # Loading image
        img = cv2.imread(put)
        height, width, channels = img.shape

        # Detecting objects#
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
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

    def rotate_image(self,mat, angle):
        """
       Функция для поворота изображений (серийный номер)
        """

        height, width = mat.shape[:2]  # форма изображения имеет 3 измерения
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D нужны координаты в обратном порядке (ширина, высота) по сравнению с формой

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # вращение вычисляет cos и sin, принимая их абсолютные значения.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # найдите новые границы ширины и высоты
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # вычтите старый центр изображения (возвращая изображение в исходное состояние) и добавьте новые координаты центра изображения.
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # поверните изображение с новыми границами и преобразованной матрицей поворота
        ser_nom = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return ser_nom

    def zero(self,n):
        return n * (n > 0)
    # вырезаем области после детекции YOLO4

    def oblasty_yolo_4(self, image, box):

        oblasty = {}
        iss = 0
        plac = 0
        # Сортируем список по у для того чтобы области шли по порядку сверху вниз
        spissok = sorted(box, reverse=False, key=lambda x: x[2])  # spiss.sort(key=custom_key)
        for l in spissok:
            cat = l[0]
            y = int(l[2])
            x = int(l[1])
            h = int(l[4])
            w = int(l[3])
            # обрезаем области и сохраняем их в словарь, добавляем к областе пиксели для увеличения области распознавания
            if ('signature' in cat) or ('photograph' in cat):
                pass  # поля подпись и фотографию не распознаем, поэтому с ними ничего не делаем
            else:
                if 'issued_by_whom' in cat:
                    ob = cat + '_' + str(iss)
                    iss += 1
                elif 'place_of_birth' in cat:
                    ob = cat + '_' + str(plac)
                    plac += 1
                elif 'series' not in cat:
                    ob = cat
                oblasty[ob] = image[self.zero(y - math.ceil(h * 0.03)):y + math.ceil(h * 1.03),
                              self.zero(x - math.ceil(w * 0.1)):x + math.ceil(w * 1.1)]
                if 'series' in cat:
                    ob = cat
                    cropped = image[self.zero(y - math.ceil(h * 0.1)):y + math.ceil(h * 1.1),
                              self.zero(x - math.ceil(w * 0.03)):x + math.ceil(w * 1.03)]
                    oblasty[ob] = self.rotate_image(cropped, 90)

        # Передаем словарь с областями на распознание

        return oblasty

    def recognition_slovar(self, oblasty):
        # __________________________________________________________________
        # задаем начальные значения
        data = {}
        data['pasport'] = []
        d = {}
        issued_by_whom = ''
        series_and_number = ''
        place_of_birth = ''
        ver = 0
        acc_ocr = 0
        col_ocr = 0
        # ________________________________________________________
        # d['ID'] = (jpg.split('.')[0]).split('/')[-1]  # записываем номер фотографии (берем имя файла

        for i, v in oblasty.items():  # цикл по всем найденым полям с их распределения по классам
            image = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)  # переводим области в серый цвет
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
                                         allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-"№ .1234567890')
            pole = ''
            # Сцепляем распознаные поля в одной области и подсчитыаем среднию увереность
            for k in range(len(result)):
                if result[k][2] * 100 >= 35:
                    pole = pole + ' ' + str(result[k][1])
                    acc_ocr += result[k][2] * 100
                    col_ocr += 1
            # ели поле не пустое то записываем результат распознавания (json +csv)
            if pole:
                pole = pole.strip()  # удаляем пробелы вконце и в неачале
                # сцепляем
                if 'issued_by_whom' in i:
                    issued_by_whom = issued_by_whom + pole + ' '
                if 'place_of_birth' in i:
                    place_of_birth = place_of_birth + pole + ' '
                if 'series_and_number' in i:
                    # print(pole)
                    if len(pole) >= 10:
                        if ver < result[k][2]:
                            series_and_number = pole
                # Убираем лишние знаки в распознание текста, если такие находятся и приводим к формату
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
                    # заменяем пол
                elif 'gender' in i:
                    if 'Е' in pole.upper() or 'Н' in pole.upper():
                        pole = 'ЖЕН.'
                    elif 'У' in pole.upper() or 'М' in pole.upper():
                        pole = 'МУЖ.'
                    d[i.split('.', 1)[0]] = pole.upper().strip()
                else:
                    d[i.split('.', 1)[0]] = pole.replace('  ', ' ').upper().strip()
        # пост обработка текста, подводим под формат паспорта
        place_of_birth = place_of_birth.upper()
        issued_by_whom = issued_by_whom.upper()
        if place_of_birth[:2] == 'C ':
            place_of_birth = place_of_birth.replace('С ', ' С. ')
        if issued_by_whom[:2] == 'C ':
            issued_by_whom = issued_by_whom.replace('С ', ' С. ')
        place_of_birth = place_of_birth.replace('ГОР ', 'ГОР. ').replace(' Г ', ' Г. ')\
            .replace('ОБЛ ','ОБЛ. ').replace('ПОС ','ПОС. ').replace(' . ', '. ')\
            .replace(' .', '.').replace('  ', ' ').replace('..', '.')
        issued_by_whom = issued_by_whom.replace('ГОР ', 'ГОР. ').replace(' С ', ' С. ')\
            .replace(' Г ', ' Г. ').replace('ОБЛ ', 'ОБЛ. ').replace('ПОС ', 'ПОС. ')\
            .replace(' . ', '. ').replace(' .', '.').replace('  ',' ').replace('..','.')
        if series_and_number:
            series_and_number = series_and_number.replace(' ', '')
            if len(series_and_number) == 10:
                series_and_number = series_and_number[:2] + ' ' + series_and_number[2:4] + ' ' + series_and_number[4:10]
            else:
                series_and_number = 'поле распознано не полностью' + series_and_number
        else:
            series_and_number = 'поле не распознано'
        # Создаем файлы json and csv
        d['issued_by_whom'] = issued_by_whom.strip()
        d['place_of_birth'] = place_of_birth.strip()
        d['series_and_number'] = series_and_number.strip()
        data['pasport'].append(d)
        with open('data.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False))
        # pprint.pprint(data['pasport'])
        return data['pasport']




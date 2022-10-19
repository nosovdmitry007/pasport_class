import cv2
import numpy as np
import easyocr
import math
import torch
import pandas as pd
class Passport:
    def __init__(self):
        self.reader = easyocr.Reader(['ru'],
                        model_storage_directory='EasyOCR/model',
                        user_network_directory='EasyOCR/user_network',
                        recog_network='custom_example',
                        gpu=False)
        self.model_round = torch.hub.load('yolov5_master', 'custom', path='yolo5/rotation.pt', source='local')
        self.model_detect = torch.hub.load('yolov5_master', 'custom', path='yolo5/detect.pt', source='local')

    def zero(self,n):
        return n * (n > 0)

    def rotate_image(self, mat, angle):
        height, width = mat.shape[:2]
        image_center = (width / 2, height / 2)
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]
        return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))


    def get_angle_rotation(self, centre, point, target_angle):
        #centre - Точка относительно которой надо вращать. tuple (x,y)
        #point - точка которую надо повернуть tuple (x,y)
        #угол куда должна повернуться point. градусы от 0 до 360.   Принцип такой: 15 часов - 0 градусов, 12 часов - 90 градусов,   9 часов - 180 градусов,  18 часов - 270 грудусов.  Отрицательных градусов быть не должно

        new_point =(point[0] - centre[0], point[1] - centre[1])  #передвигаем центр системы координат в точку центр. у нее будет (0,0) ищем новые координаты у точки point
        a,b = new_point[0], new_point[1]
        res = math.atan2(b,a) #ищем полярный угол у new_point
        if (res < 0) :
              res += 2 * math.pi
        return (math.degrees(res)+target_angle) % 360  #возвращаем угол поворота для cv2

    def result(self,img):
        return self.model_round(img)

    def get_image_after_rotation(self, img):
        results = self.result(img)
        pd = results.pandas().xyxy[0]
        pd = pd.assign(centre_x = pd.xmin + (pd.xmax-pd.xmin)/2)
        pd = pd.assign(centre_y = pd.ymin + (pd.ymax-pd.ymin)/2)
        tmp = pd.loc[pd['name']=='niz']
        N, V = None, None
        for index, row in tmp.iterrows():
            N = (row['centre_x'], row['centre_y'])
            break
        #получим координаты верха, там где печать
        tmp = pd.loc[pd['name']=='verh']
        for index, row in tmp.iterrows():
            V = (row['centre_x'], row['centre_y'])
            break
        if N == None or V == None: #похоже там нет нужных нам строк
            return img

        angle = self.get_angle_rotation(N, V, 90)
        img = self.rotate_image(img, angle)  #вращаем той процедурой, что выше
        return img

    def crop_img(self, img):
        results = self.result(img)
        pd = results.pandas().xyxy[0]
        #определяем координаты вырезки
        x1 =int(pd.xmin.min())
        x2 = int(pd.xmax.max())
        y1 = int(pd.ymin.min())
        y2 = int(pd.ymax.max())
        img = img[y1:y2,x1:x2]
        return img

    def get_crop(self,file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.get_image_after_rotation(image)
        image = self.get_image_after_rotation(image) #второй подряд поворот еще лучше выравнивает.
        image = self.crop_img(image)

        return image

    def yolo_5(self, img):
        results = self.model_detect(img)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.7)[0])
        # print(df)
        ob = pd.DataFrame()
        ob['class'] = df['name']
        ob['x'] = df['xmin']
        ob['y'] = df['ymin']
        ob['w'] = df['xmax']-df['xmin']
        ob['h'] = df['ymax']-df['ymin']
        oblasty = ob.values.tolist()
        return img, oblasty

    def oblasty_yolo_5(self, image, box):
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
                    oblasty[ob] = image[self.zero(y - math.ceil(h * 0.07)):y + math.ceil(h * 1.1),
                                  self.zero(x - math.ceil(w * 0.1)):x + math.ceil(w * 1.1)]
                if 'series' in cat:
                    ob = cat
                    cropped = image[self.zero(y - math.ceil(h * 0.1)):y + math.ceil(h * 1.1),
                              self.zero(x - math.ceil(w * 0.03)):x + math.ceil(w * 1.03)]
                    oblasty[ob] = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE) #self.rotate_image(cropped, 90)
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
                                        allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-"№ .1234567890')
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
                    pole1 = ''
                    if 'Е' in pole.upper() or 'Н' in pole.upper() or 'ЕН' in pole.upper():
                        pole1 = 'ЖЕН.'
                    if 'У' in pole.upper() or 'М' in pole.upper() or 'УЖ' in pole.upper():
                        pole1 = 'МУЖ.'
                    pole = pole1
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
            series_and_number = ''
        d['issued_by_whom'] = issued_by_whom.strip()
        d['place_of_birth'] = place_of_birth.strip()
        d['series_and_number'] = series_and_number.strip()
        data['pasport'].append(d)
        return data['pasport']

    def detect_passport(self,photo,povorot):
        pole = ['date_of_birth','date_of_issue','first_name','gender','issued_by_whom',
                'patronymic','place_of_birth','series_and_number','surname','unit_code']
        if povorot == 0:
            # res = self.result(photo)
            croped = self.get_crop(photo)
            if croped != '':
                img, detect = self.yolo_5(croped)
                obl = self.oblasty_yolo_5(img, detect)
                rec = self.recognition_slovar(obl)
                key = list(rec[0].keys())
                value = list(rec[0].values())
                if set(key) == set(pole):
                    if '' in value:
                        flag = 1
                    else:
                        flag = 0
                else:
                    flag = 1
                return rec[0], flag
            else:
                rec = {}
                return rec, 1
        else:
            croped = cv2.imread(photo,cv2.COLOR_BGR2GRAY)
            img, detect = self.yolo_5(croped)
            if detect != '':
                obl = self.oblasty_yolo_5(img, detect)
                rec = self.recognition_slovar(obl)
                key = list(rec[0].keys())
                value = list(rec[0].values())
                if set(key) == set(pole):
                    if '' in value:
                        flag = 1
                    else:
                        flag = 0
                else:
                    flag = 1
                return rec[0], flag
            else:
                rec = {}
                return rec, 1

class INN(Passport):
    def __init__(self):
        self.reader = easyocr.Reader(['ru'],
                        model_storage_directory='EasyOCR/model',
                        user_network_directory='EasyOCR/user_network',
                        recog_network='custom_example',
                        gpu=False)
        self.model_round_inn = torch.hub.load('yolov5_master', 'custom', path='yolo5/inn_rotation.pt', source='local')
        self.model_detect_inn = torch.hub.load('yolov5_master', 'custom', path='yolo5/fio_INN.pt', source='local')
    def result_inn(self,img):
        return self.model_round_inn(img)
    def get_image_after_rotation_inn(self, img):
        results = self.result_inn(img)
        pd = results.pandas().xyxy[0]
        pd = pd.assign(centre_x = pd.xmin + (pd.xmax-pd.xmin)/2)
        pd = pd.assign(centre_y = pd.ymin + (pd.ymax-pd.ymin)/2)
        tmp = pd.loc[pd['name']=='niz']
        N, V = None, None
        for index, row in tmp.iterrows():
            N = (row['centre_x'], row['centre_y'])
            break
        #получим координаты верха, там где печать
        tmp = pd.loc[pd['name']=='verh']
        for index, row in tmp.iterrows():
            V = (row['centre_x'], row['centre_y'])
            break
        if N == None or V == None: #похоже там нет нужных нам строк
            return img

        angle = self.get_angle_rotation(N, V, 90)
        img = self.rotate_image(img, angle)  #вращаем той процедурой, что выше
        return img

    def crop_img_inn(self, img):
        results = self.result_inn(img)
        pd = results.pandas().xyxy[0]
        x1 =int(pd.xmin.min())
        x2 = int(pd.xmax.max())
        y1 = int(pd.ymin.min())
        y2 = int(pd.ymax.max())
        img = img[y1:y2,x1:x2]
        return img

    def get_crop_inn(self,file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.get_image_after_rotation_inn(image)
        image = self.get_image_after_rotation_inn(image) #второй подряд поворот еще лучше выравнивает.

        return image
    def yolo_5_inn(self, img):
        results = self.model_detect_inn(img)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.7)[0])
        ob = pd.DataFrame()
        ob['class'] = df['name']
        ob['x'] = df['xmin']
        ob['y'] = df['ymin']
        ob['w'] = df['xmax']-df['xmin']
        ob['h'] = df['ymax']-df['ymin']
        oblasty = ob.values.tolist()
        return img, oblasty

    def oblasty_yolo_5_inn(self, image, box):
        oblasty = {}
        spissok = sorted(box, reverse=False, key=lambda x: x[2])
        for l in spissok:
            cat = l[0]
            y = int(l[2])
            x = int(l[1])
            h = int(l[4])
            w = int(l[3])
            ob = cat
            oblasty[ob] = image[self.zero(y - math.ceil(h * 0.1)):y + math.ceil(h * 1.1),
                      self.zero(x - math.ceil(w * 0.03)):x + math.ceil(w * 1.03)]
        return oblasty

    def recognition_slovar_inn(self, oblasty):
        data = {}
        data['inn'] = []
        d = {}
        for i, v in oblasty.items():
            image = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
            if 'inn' in i:
                result = self.reader.readtext(image, allowlist='0123456789-. ')
            elif 'fio' in i:
                result = self.reader.readtext(image,
                                         allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-')
            pole = ''
            for k in range(len(result)):
                pole = pole + ' ' + str(result[k][1])
            if pole:
                pole = pole.strip()
                d[i.split('.', 1)[0]] = pole.upper().strip()
        data['inn'].append(d)
        return data['inn']

    def detect_inn(self,photo,povorot):
        pole = ['fio','inn']
        if povorot == 0:
            croped = self.get_crop_inn(photo)
            if croped != '':
                img, detect = self.yolo_5_inn(croped)
                obl = self.oblasty_yolo_5_inn(img, detect)
                rec = self.recognition_slovar_inn(obl)
                key = list(rec[0].keys())
                value = list(rec[0].values())
                if set(key) == set(pole):
                    if '' in value:
                        flag = 1
                    else:
                        flag = 0
                else:
                    flag = 1
                return rec[0], flag
            else:
                rec = {}
                return rec, 1
        else:
            croped = cv2.imread(photo,cv2.COLOR_BGR2GRAY)
            img, detect = self.yolo_5_inn(croped)
            if detect != '':
                obl = self.oblasty_yolo_5_inn(img, detect)
                rec = self.recognition_slovar_inn(obl)
                key = list(rec[0].keys())
                value = list(rec[0].values())
                if set(key) == set(pole):
                    if '' in value:
                        flag = 1
                    else:
                        flag = 0
                else:
                    flag = 1
                return rec[0], flag
            else:
                rec = {}
                return rec, 1

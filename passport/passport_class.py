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

    @staticmethod
    def save_bounding_boxes(result_image) -> None:
        # boxes will save into runs/detect/exp
        result_image.crop(save=True)
        return None

    def zero(self, n):
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
        new_point =(point[0] - centre[0], point[1] - centre[1])
        a,b = new_point[0], new_point[1]
        res = math.atan2(b,a)
        if (res < 0) :
              res += 2 * math.pi
        return (math.degrees(res)+target_angle) % 360

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
        tmp = pd.loc[pd['name']=='verh']
        for index, row in tmp.iterrows():
            V = (row['centre_x'], row['centre_y'])
            break
        if N == None or V == None:
            return img

        angle = self.get_angle_rotation(N, V, 90)
        img = self.rotate_image(img, angle)
        return img

    def crop_img(self, img):
        results = self.result(img)
        pd = results.pandas().xyxy[0]
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
        image = self.get_image_after_rotation(image)
        image = self.crop_img(image)
        return image

    def yolo_5(self, img):
        results = self.model_detect(img)
        self.save_bounding_boxes(results)
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
            # if image.shape[0] <=23:
            #     image = self.resiz(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Для каждого класса устанавливаем свои ограничения на распознания классов

            if 'date' in i:
                result = self.reader.readtext(image, allowlist='0123456789.')
            elif 'code' in i:
                result = self.reader.readtext(image, allowlist='0123456789-')
            elif 'series' in i:
                result = self.reader.readtext(image, allowlist='0123456789 ')
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
                if result[k][2] * 100 >= 25:
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
                                        .replace('  ', ' ')\
                                        .replace('--', '-')

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
                                        .replace('  ', ' ')\
                                        .replace('--', '-')
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

    def detect_passport(self,photo):
        pole = ['date_of_birth','date_of_issue','first_name','gender','issued_by_whom',
                'patronymic','place_of_birth','series_and_number','surname','unit_code']

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
                    if len(rec[0]['date_of_birth']) != 10 or len(rec[0]['date_of_issue']) != 10 or len(rec[0]['series_and_number']) != 12 or len(rec[0]['unit_code']) != 7:
                        flag = 1
            else:
                flag = 1

            if flag == 1:
                rec =rec[0]# {}
            else:
                rec = rec[0]
            return rec, flag
        else:
            return {}, 1


class INN(Passport):
    def __init__(self):
        self.reader = easyocr.Reader(['ru'],
                                     model_storage_directory='EasyOCR/model',
                                     user_network_directory='EasyOCR/user_network',
                                     recog_network='custom_example',
                                     gpu=False)

        self.model_round_inn = torch.hub.load('yolov5_master', 'custom', path='yolo5/inn_rotation.pt', source='local')
        self.model_detect_inn = torch.hub.load('yolov5_master', 'custom', path='yolo5/fio_INN.pt', source='local')
        self.model_numbers = torch.hub.load('yolov5_master', 'custom', path='yolo5/best_X_INN.pt', source='local')

    def result_inn(self, img):
        return self.model_round_inn(img)

    def get_angle_rotation_inn(self, centre, point, target_angle):
        new_point = (point[0] - centre[0], point[1] - centre[
            1])  # передвигаем центр системы координат в точку центр. у нее будет (0,0) ищем новые координаты у точки point
        a, b = new_point[0], new_point[1]
        res = math.atan2(b, a)  # ищем полярный угол у new_point
        if (res < 0):
            res += 2 * math.pi
        return (math.degrees(res) + target_angle) % 360  # возвращаем угол поворота для cv2

    def get_image_after_rotation_inn(self, img, point1, point2):
        results = self.result_inn(img)
        pd = results.pandas().xyxy[0]
        pd = pd.assign(centre_x=pd.xmin + (pd.xmax - pd.xmin) / 2)
        pd = pd.assign(centre_y=pd.ymin + (pd.ymax - pd.ymin) / 2)
        tmp = pd.loc[pd['name'] == point1]
        LEFT, RIGHT = None, None
        for index, row in tmp.iterrows():
            LEFT = (row['centre_x'], row['centre_y'])
            break
        # получим координаты верха, там где печать
        tmp = pd.loc[pd['name'] == point2]
        for index, row in tmp.iterrows():
            RIGHT = (row['centre_x'], row['centre_y'])
            break
        if LEFT == None or RIGHT == None:  # похоже там нет нужных нам строк
            return img

        angle = self.get_angle_rotation_inn(LEFT, RIGHT, 0)
        img = self.rotate_image(img, angle)  # вращаем той процедурой, что выше
        return img

    def get_crop_inn(self, file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # сразу в серый
        image = self.get_image_after_rotation_inn(image, point1='sv',
                                                  point2='vo')  # передаем в процедуру метки, котоыре будут использоваться для вращения первая - левая метка, вторая правая, будут ставиться в горизонт
        image = self.get_image_after_rotation_inn(image, point1='lev', point2='prav')
        return image

    def yolo_5_inn(self, img):
        results = self.model_detect_inn(img)
        self.save_bounding_boxes(results)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.7)[0])
        ob = pd.DataFrame()
        ob['class'] = df['name']
        ob['x'] = df['xmin']
        ob['y'] = df['ymin']
        ob['w'] = df['xmax'] - df['xmin']
        ob['h'] = df['ymax'] - df['ymin']
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

    def Intersection(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return interArea

    def IoU(self, boxA, boxB):
        # compute the area of intersection rectangle
        interArea = self.Intersection(boxA, boxB)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # calculation iou
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def numbers_detect(self, img):

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = self.model_numbers(image)

        final_res = result.pandas().xyxy[0]
        result = result.pandas().xyxy[0]

        # delete intersection objects
        for obj1 in range(len(result)):
            for obj2 in range(len(result)):

                boxA = result.iloc[obj1, :4].values
                boxB = result.iloc[obj2, :4].values

                if boxA.all() != boxB.all():
                    if self.IoU(boxA, boxB) > 0.2:
                        if result.iloc[obj1, 4] > result.iloc[obj2, 4]:
                            final_res = final_res[final_res.xmin != result.iloc[obj2, 0]]

                        else:
                            final_res = final_res[final_res.xmin != result.iloc[obj1, 0]]
        return final_res

    def recognition_slovar_inn(self, oblasty):
        data = {}
        data['inn'] = []
        d = {}
        for i, v in oblasty.items():
            image = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)

            # plt.imshow(image)
            # plt.show()

            if 'inn' in i:

                numbs = self.numbers_detect(image).sort_values(by=['xmin']).iloc[:, 5].values
                res_numbs = ''.join(str(e) for e in numbs)

                result = [([], f'{res_numbs}', 0.0)]

            elif 'fio' in i:
                result = self.reader.readtext(image,
                                              allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ- ')

            pole = ''
            for k in range(len(result)):
                pole = pole + ' ' + str(result[k][1])
            if pole:
                pole = pole.strip()
                d[i.split('.', 1)[0]] = pole.upper().strip()
        data['inn'].append(d)
        return data['inn']

    def detect_inn(self, photo):
        pole = ['fio', 'inn']
        croped = self.get_crop_inn(photo)

        if croped != '':
            img, detect = self.yolo_5_inn(croped)

            for l in detect:
                x = l[1]
                y = l[2]
                w = l[3]
                h = l[4]

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


class Snils:
    def __init__(self):
        self.reader = easyocr.Reader(['ru'],
                        model_storage_directory='EasyOCR/model',
                        user_network_directory='EasyOCR/user_network',
                        recog_network='custom_example',
                        gpu=False)

        self.model_round = torch.hub.load('yolov5_master', 'custom', path='yolo5/snils_rotation.pt', source='local')
        self.model_detect = torch.hub.load('yolov5_master', 'custom', path='yolo5/snils_detect.pt', source='local')
        self.model_numbers = torch.hub.load('yolov5_master', 'custom', path='yolo5/yolov5m.pt', source='local')

    @staticmethod
    def save_bounding_boxes(result_image) -> None:
        # boxes will save into runs/detect/exp
        result_image.crop(save=True)
        return None

    def rotate_image(self, mat, angle, point):
        height, width = mat.shape[:2]
        image_center = (width/2, height/2)
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0])
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    def get_angle_rotation(self, centre, point, target_angle):
        new_point =(point[0] - centre[0], point[1] - centre[1])
        a,b = new_point[0], new_point[1]
        res = math.atan2(b,a)
        if (res < 0) :
            res += 2 * math.pi
        return (math.degrees(res)+target_angle) % 360

    def get_image_after_rotation(self, img):
        results = self.model_round(img)
        ('get_image_after_rotation',results)
        pd = results.pandas().xyxy[0]
        pd = pd.assign(centre_x = pd.xmin + (pd.xmax-pd.xmin)/2)
        pd = pd.assign(centre_y = pd.ymin + (pd.ymax-pd.ymin)/2)

        tmp = pd.loc[pd['name']=='Svidetelstvo_test']

        N, V = None, None
        for index, row in tmp.iterrows():
            N = (row['centre_x'], row['centre_y'])
            break
        #получим координаты верха, там где печать
        tmp = pd.loc[pd['name']=='Svidetelstvo_train']
        for index, row in tmp.iterrows():
            V = (row['centre_x'], row['centre_y'])
            break
        if N == None or V == None: #похоже там нет нужных нам строк
            return img

        angle = self.get_angle_rotation(N, V, 180)

        img = self.rotate_image(img, angle, N)  #вращаем той процедурой, что выше
        return img

    def crop_img(self, img):
        results = self.model_round(img)
        pd = results.pandas().xyxy[0]
        try:
        #определяем координаты вырезки
            x1 =int(pd.xmin.min())
            x2 = int(pd.xmax.max())

            y1 = int(pd.ymin.min())
            y2 = int(pd.ymax.max())

            img = img[y1:y2,x1:x2]
        except Exception:
            print(pd)
        return img

    def get_crop(self, file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = self.get_image_after_rotation(image)
        image = self.get_image_after_rotation(image) #второй подряд поворот еще лучше выравнивает

        return image

    def zero(self,n):
        return n * (n > 0)

    def yolo_5_snils(self, img):
        results = self.model_detect(img)
        self.save_bounding_boxes(results)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.6)[0])
        ob = pd.DataFrame()
        ob['class'] = df['name']
        ob['x'] = df['xmin']
        ob['y'] = df['ymin']
        ob['w'] = df['xmax']-df['xmin']
        ob['h'] = df['ymax']-df['ymin']
        oblasty = ob.values.tolist()
        return img, oblasty

    def oblasty_yolo_5_snils(self, image, box):
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

    def Intersection(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return interArea

    def IoU(self, boxA, boxB):
        # compute the area of intersection rectangle
        interArea = self.Intersection(boxA, boxB)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # calculation iou
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def numbers_detect(self, img):

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = self.model_numbers(image)

        final_res = result.pandas().xyxy[0]
        result = result.pandas().xyxy[0]

        # delete intersection objects
        for obj1 in range(len(result)):
            for obj2 in range(len(result)):

                boxA = result.iloc[obj1, :4].values
                boxB = result.iloc[obj2, :4].values

                if boxA.all() != boxB.all():
                    if self.IoU(boxA, boxB) > 0.2:
                        if result.iloc[obj1, 4] > result.iloc[obj2, 4]:
                            final_res = final_res[final_res.xmin != result.iloc[obj2, 0]]

                        else:
                            final_res = final_res[final_res.xmin != result.iloc[obj1, 0]]
        return final_res


    def recognition_slovar_snils(self, oblasty):
        data = {}
        data['snils'] = []
        d = {}
        fio=''
        for i, v in oblasty.items():
            image = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)

            # plt.imshow(image)
            # plt.show()

            if 'number_strah' in i:
                # result = self.reader.readtext(image, allowlist='0123456789- ')

                numbs = self.numbers_detect(image).sort_values(by=['xmin']).iloc[:, 5].values
                res_numbs = ''.join(str(e) for e in numbs)
                res_numbs = f'{res_numbs[:3]}-{res_numbs[3:6]}-{res_numbs[6:9]}_{res_numbs[-2:]}'
                result = [([], f'{res_numbs}', 0.0)]

            elif 'fio' in i:
                result = self.reader.readtext(image,
                                         allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ- ')
            pole = ''
            for k in range(len(result)):
                pole = pole + ' ' + str(result[k][1]).replace(' ', '').replace('_', ' ')
            if pole:
                pole = pole.strip()
                if 'fio' in i:
                    fio = fio + ' ' + pole.upper().strip()
                else:
                    d[i.split('.', 1)[0]] = pole.upper().strip()
        d['fio'] = fio.strip()
        data['snils'].append(d)
        return data['snils']

    def detect_snils(self,photo):
        pole = ['number_strah','fio1','fio2','fio3']

        croped = self.get_crop(photo)
        if croped != '':
            img, detect = self.yolo_5_snils(croped)
            obl = self.oblasty_yolo_5_snils(img, detect)
            rec = self.recognition_slovar_snils(obl)
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


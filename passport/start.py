import pprint
from passport_class import passport

z = passport()
img,detect = z.yolo_4(r'C:\Users\nosov\PycharmProjects\pasport_class\passport\7653.jpg')
obl = z.oblasty_yolo_4(img,detect)
rec = z.recognition_slovar(obl)
pprint.pprint(rec)
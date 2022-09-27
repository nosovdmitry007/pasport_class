import pprint
from passport_class import passport

z = passport()
crop = z.yolo_4_round('baza/1.jpg')
aut = z.auto_rotait(crop)
img,detect = z.yolo_4(aut)
obl = z.oblasty_yolo_4(img,detect)
rec = z.recognition_slovar(obl)
pprint.pprint(rec)

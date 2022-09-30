# Система распознавания данных по фото документов

## Установка:

1. Клонируем репозиторий GIT:
```
    git clone https://github.com/nosovdmitry007/pasport_class.git -b add_auto_rotation
```
2. Устанавливаем необхоимые библиотеки:
```
    pip install --upgrade pip
    cd passport
    pip install -r requirements.txt
```
## Пример распознавания паспорта:
1. Создаём экземпляр класса:
```
   from passport_class import Passport
   psprt = Passport()
```
2. Запускаем распознавание данных:
```
   result = psprt.detect_passport('path_to_image.jpg')
```
На выходе получаем данные в виде словаря и флага.

```
result = ({'date_of_birth':'Дата рождения',
      'date_of_issue':'Дата выдачи',
      'first_name':'Имя',
      'gender':'Пол',
      'issued_by_whom':'Кем выдан',
      'patronymic':'Отчество',
      'place_of_birth':'Место рождение',
      'series_and_number':'Серия и номер',
      'surname':'Фамилия',
      'unit_code':'Код подразделения'},
      0)
```
Флаг = 1, паспорт не обнаружен, или хотя бы 1 поле не распознано

Флаг = 0, все поля паспорта распознаны 

В будущем в модуль будут добавлены классы для распознания СНИЛС и ИНН.

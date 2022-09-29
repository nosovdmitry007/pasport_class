Установка:

1. клонируем репозиторий GIT:

    `git clone https://github.com/nosovdmitry007/pasport_class.git -b Add_auto_rotation`

2. Устанавливаем необхоимые библиотеки:

    `pip install --upgrade pip`

    `pip install -r passport/requirements.txt`

Пример запуска модуля для распознания паспорта:
1. заходим в файл start.py
2. в строке

    `pprint.pprint(passport().detect_passport('[yuor photo]'))`

меняем `[yuor photo] ` на путь к фотографии с паспортом

В будущем в модуль будут добавлены классы для распознания СНИЛС и ИНН.

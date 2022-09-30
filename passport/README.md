Установка:

1. клонируем репозиторий GIT:

    `git clone https://github.com/nosovdmitry007/pasport_class.git -b Add_auto_rotation`

2. Устанавливаем необхоимые библиотеки:

    `pip install --upgrade pip`
   
   `cd passport`

    `pip install -r requirements.txt`

Пример распознавания паспорта:
1. Создаём экземпляр класса:

   `from passport_class import Passport`

   `psprt = Passport()`
2. Запускаем распознавание данных:

   `result = psprt.detect_passport('path_to_image.jpg')`


В будущем в модуль будут добавлены классы для распознания СНИЛС и ИНН.

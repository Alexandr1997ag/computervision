# computervision

Абстрактный класс FoodCoords имеет абстрактный метод go_network

реализовано 2 сценария:

1) воспользовавшись библиотекой opencv производится детектирование предмета на картинке. Важно! картинка должна быть без шума. (tZ7mr.png - использована картинка) Нейросеть здесь не используется.

2) реализована нейросеть решающая задачу небинарной классификации (5 классов цветов). Производится изначально обучение, смотрим лос и accuracy, далее дообучаем ее. 15 эпох.
92/92 [==============================] - 32s 344ms/step - loss: 0.5209 - accuracy: 0.8014 - val_loss: 0.6448 - val_accuracy: 0.7507
This image most likely belongs to sunflowers with a 90.64 percent confidence.  
Подавал картинку (https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg)).

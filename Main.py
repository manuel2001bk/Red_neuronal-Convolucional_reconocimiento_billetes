
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = []
historial = []


def metodo2():
    data_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        validation_split=0.15
    )
    data_gen_entrenamiento = data_gen.flow_from_directory(
        f'Dataset',
        target_size=(150, 200),
        batch_size=16,
        shuffle=True,
        subset='training')
    data_gen_pruebas = data_gen.flow_from_directory(
        f'Dataset',
        target_size=(150, 200),
        batch_size=16,
        shuffle=True,
        subset='validation')
    plt.figure(figsize=(20, 20))
    for imagen, etiqueta in data_gen_entrenamiento:
        for i in range(10):
            print(etiqueta[i])
            plt.subplot(2, 5, i+1)
            plt.imshow((imagen[i]/255))
        break
    plt.show()
    plt.figure(figsize=(20, 20))

    for imagen, etiqueta in data_gen_pruebas:
        for i in range(10):
            print(etiqueta[i])
            plt.subplot(2, 5, i+1)
            plt.imshow(imagen[i]/255)
        break
    plt.show()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(150, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    historial = model.fit(
        data_gen_entrenamiento,
        epochs=50,
        validation_data=data_gen_pruebas,
    )
    model.save('modelo.h5')
    plt.plot(historial.history['loss'])
    plt.show()


if __name__ == "__main__":
    metodo2()

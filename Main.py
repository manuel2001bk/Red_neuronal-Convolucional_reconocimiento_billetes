
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = []
historial = []



def metodo2():
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=15,
        zoom_range=[0.5, 1.5],
        horizontal_flip=True,
        validation_split=0.2
    )
    data_gen_entrenamiento = data_gen.flow_from_directory(
        f'Dataset',
        target_size=(400, 600),
        batch_size=32,
        shuffle=True,
        subset='training')
    data_gen_pruebas = data_gen.flow_from_directory(
        f'Dataset',
        target_size=(400, 600),
        batch_size=32,
        shuffle=True,
        subset='validation')
    for imagen, etiqueta in data_gen_entrenamiento:
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(imagen[i])
        break
    plt.show()
    for imagen, etiqueta in data_gen_pruebas:
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(imagen[i])
        break
    plt.show()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=(400, 600, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
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
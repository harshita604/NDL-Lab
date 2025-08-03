import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf

def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train= (X_train.astype(np.float32)/255.0 )
    X_test= (X_test.astype(np.float32)/255.0 )

    X_train= np.expand_dims(X_train, axis=-1)
    X_test= np.expand_dims(X_test, axis=-1)

    y_train= to_categorical(y_train, num_classes=10)
    y_test= to_categorical(y_test, num_classes=10)

    return (X_train, y_train), (X_test, y_test)

def create_LeNet():
    model= Sequential()
    model.add(Conv2D(6, kernel_size=5, activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(16, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

def create_AlexNet():          #modified to improve computational efficiency
    model= Sequential()
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3,padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

def train(model_fn, train_data, test_data, model_name):
    model= model_fn()
    optimizer= Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    
    X_train, y_train= train_data
    X_test, y_test= test_data
    print(f"Training for {model_name}: ")
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    print(f"Loss and accuracy for {model_name}")
    loss, accuracy= model.evaluate(X_test, y_test)
    print(loss)
    print(accuracy)
    return model

def visualize_filters(model):
    first_conv= None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            first_conv= layer
            break
    if first_conv is not None:
        weights= first_conv.get_weights()[0]
        num_filters= min(6, weights.shape[-1])
        fig, axes= plt.subplots(1, num_filters, figsize=(10,5))
        for i in range(num_filters):
            filter_img= weights[:,:,0,i]
            axes[i].imshow(filter_img, cmap='gray')
            axes[i].axis('off')
    plt.suptitle("First conv layer filters")
    plt.show()



models=[
    (create_LeNet, "LeNet"),
    (create_AlexNet,"AlexNet")
]

train_data, test_data= get_data()

for model_fn, model_name in models:
    trained= train(model_fn, train_data, test_data, model_name)
    visualize_filters(trained)



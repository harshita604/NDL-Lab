import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import tensorflow as tf
tf.config.run_functions_eagerly(True)

(x_train, y_train), (x_test, y_test)= mnist.load_data()

x_train= np.expand_dims(x_train, axis=-1)
x_test= np.expand_dims(x_test, axis=-1)

x_train= np.repeat(x_train, 3, axis=-1)
x_test= np.repeat(x_test, 3, axis=-1)

x_train_resize= np.array([cv2.resize(img, (32,32)) for img in x_train])
x_test_resize= np.array([cv2.resize(img, (32,32)) for img in x_test])

x_train= x_train/255.0
x_test= x_test/255.0
x_train_resize= x_train_resize/255.0
x_test_resize= x_test_resize/255.0

y_train= to_categorical(y_train, num_classes=10)
y_test= to_categorical(y_test, num_classes=10)

base_vgg= VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
for layer in base_vgg.layers:
    layer.trainable = False

x= Flatten()(base_vgg.output)
x= Dense(256, activation='relu')(x)
x= Dense(10, activation='softmax')(x)

model= Model(inputs= base_vgg.input, outputs=x)
optimizer= Adam(learning_rate= 0.001)
model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_resize, y_train, epochs=5, batch_size=32, validation_data=(x_test_resize, y_test))
print(f"For VGG16:")

loss, accuracy= model.evaluate(x_test_resize, y_test)
print(loss)
print(accuracy)

input_layer=Input(shape=(28,28,3))

x= Conv2D(64, (3,3), padding='same', activation='relu')(input_layer)
x= MaxPooling2D((2,2))(x)
x= Conv2D(128, (3,3), padding='same', activation='relu')(x)
x=MaxPooling2D((2,2))(x)
x= Flatten()(x)
x= Dense(256, activation='relu')(x)
x= Dense(10, activation='softmax')(x)
optimizer= Adam()
model1= Model(inputs=input_layer, outputs=x)
model1.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

print(f"For Places Net:")
model1.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
loss, accuracy= model1.evaluate(x_test, y_test)
print(loss)
print(accuracy)


def plot_confusion_matrix(model, y_test, x_test):
    y_pred= np.argmax(model.predict(x_test))
    cm= confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

plot_confusion_matrix(model, x_test_resize, y_test)
plot_confusion_matrix(model1, x_test, y_test)
def visualise_maps(model, x_samples):
    layer_outputs= [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
    activation_model= Model(inputs= model.input, outputs= layer_outputs)
    activations= activation_model.predict(np.expand_dims(x_samples, axis=0))
    for i, activation in range(activations[:3]):
        plt.figure(figsize=(10,5))

        for j in range(min(6, activation.shape[-1])):
            plt.subplot(1,6,j+1)
            img= activation[0,:,:,j]
            plt.imshow(img, cmap='virdis')
        plt.show()

visualise_maps(model, x_test_resize[0])
visualise_maps(model1, x_test[0])

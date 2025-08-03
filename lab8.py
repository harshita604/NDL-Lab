import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test)= mnist.load_data()
X_train= X_train.astype(np.float32)/255.0
X_test= X_test.astype(np.float32)/255.0

X_train= np.expand_dims(X_train, axis=-1)
X_test= np.expand_dims(X_test, -1)

y_train= to_categorical(y_train, num_classes=10)
y_test= to_categorical(y_test, num_classes=10)

input_shape= X_train.shape[1:]
model= Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history= model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy= model.evaluate(X_test, y_test)
print(loss)
print(accuracy)

acc= history.history['accuracy']
val_acc= history.history['val_accuracy']

plt.plot(acc)
plt.plot(val_acc)
plt.title("Accuracy vs Epochs")
plt.ylabel("Accuracy")
plt.xlabel("Epocjs")
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()          
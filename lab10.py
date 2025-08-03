from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adagrad, Adam, RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.run_functions_eagerly(True)

def get_data():
    (X_train, y_train), (X_test, y_test)= mnist.load_data()

    X_train= X_train/255.0
    X_test= X_test/255.0

    y_train= to_categorical(y_train, num_classes=10)
    y_test= to_categorical(y_test, num_classes=10)

    return (X_train, y_train), (X_test, y_test)

def create_model():
    model= Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def train(optimizer, name, train_data, test_data):
    model= create_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    X_train, y_train= train_data
    X_test, y_test= test_data

    print(f"Training with {name} optmizer")
    hist= model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data= (X_test, y_test), verbose=2)

    print(f"Loss and accuracy with {name}:")
    loss, accuracy= model.evaluate(X_test, y_test)
    print(loss)
    print(accuracy)
    return hist.history['accuracy']


train_data, test_data= get_data()

optimizers=[
    (Adagrad(learning_rate=0.005), 'AdaGrad'),
    (RMSprop(learning_rate=0.001), 'RMSProp'),
    (Adam(learning_rate=0.001), 'Adam')
]

accuracies={}
for opt, name in optimizers:
    accuracy= train(opt, name, train_data, test_data)
    accuracies[name]= accuracy

plt.figure(figsize=(10, 6))
for name, acc in accuracies.items():
    plt.plot(acc, label=name)
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop

rescale_val = 1./255
learn_rate = .0001
batch_size = 32
classes = 10
train_vals = 50
test_vals = 10
epochs = 30

def reshape_set(data_set):
    return data_set.reshape(data_set.shape[0], -1, 1).astype('float32')/255

def create_sequential_model(size):
    model = Sequential()
    model.add(SimpleRNN(100,
                    kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    recurrent_initializer=initializers.Identity(gain=0.5),
                    activation='relu',
                    input_shape=size))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    (train_a, train_b), (test_a, test_b) = mnist.load_data()
    train_a = reshape_set(train_a)
    test_a = reshape_set(test_a)

    train_b = keras.utils.to_categorical(train_b, classes)
    test_b = keras.utils.to_categorical(test_b, classes)

    seq_model = create_sequential_model(train_a.shape[1:])

    optimizer_prop = RMSprop(lr=learn_rate)
    seq_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_prop,
              metrics=['accuracy'])

    seq_model.fit(train_a, train_b,
          batch_size=batch_size,
          verbose=1,
          epochs=epochs,
          validation_data=(test_a, test_b))

    scores = seq_model.evaluate(test_a, test_b, verbose=0)
    print('RNN test score:', scores[0])
    print('RNN test accuracy:', scores[1])
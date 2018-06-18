import os
import numpy as np
from keras import backend
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import RMSprop

def get_image_size():
    first_image = load_img('data/train/noPneumonia/IM-0001-0001.jpg')
    first_image_arr = img_to_array(first_image)
    return first_image_arr.shape

def create_generator(generator, directory, size, batch):
    return generator.flow_from_directory(
        directory,
        target_size=size,
        batch_size=batch,
        class_mode='binary')

def create_sequential_model(image_size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=image_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

rescale_val = 1./255
shear_rge = 0.15
zoom_rge = 0.15
learn_rate = .0001
aug_batch_size = 16
test_batch_size = 32
train_vals = 100
test_vals = 40
epochs = 15
if __name__ == '__main__':
    image_size = get_image_size()
    image_gen = ImageDataGenerator(rescale=rescale_val)
    image_aug_gen = ImageDataGenerator(rescale=rescale_val, shear_range=shear_rge, zoom_range=zoom_rge, horizontal_flip=True,fill_mode='nearest')
    t_gen = create_generator(image_aug_gen, 'data/train/', (image_size[0], image_size[1]), aug_batch_size)
    v_gen = create_generator(image_gen, 'data/validation/', (image_size[0], image_size[1]), test_batch_size)
    seq_model = create_sequential_model(image_size)
    optimizer_prop = RMSprop(lr=learn_rate)
    seq_model.compile(loss='binary_crossentropy',
              optimizer=optimizer_prop,
              metrics=['accuracy'])
    seq_model.fit_generator(
        t_gen,
        steps_per_epoch=train_vals/aug_batch_size,
        nb_epoch=epochs,
        validation_data=v_gen,
        validation_steps=test_vals)
    print("CNN Model Accuracy: " + str(seq_model.evaluate_generator(v_gen, test_vals)[1] * 100) + "%")

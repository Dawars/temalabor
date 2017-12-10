import keras
import os
from keras.callbacks import ModelCheckpoint, TensorBoard

from load_hand_data import load_data, shuffle_data, preprocess_label, preprocess_feature

from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras import models
from keras import layers

model_name = "cv2_batch_8_epoch_10000_data_50_adam_random_inception"

# loading datasets
train_X, train_Y = load_data(['Train1'], ['female'], read_labels=True)
train_X = preprocess_feature(train_X)
train_Y = preprocess_label(train_Y)

print("train x {} train y {}".format(train_X.shape, train_Y.shape))

# building model

conv_base = InceptionV3(weights=None,
                        input_shape=(224, 224, 3),
                        include_top=False)

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='selu'))
model.add(layers.Dense(128, activation='selu'))
model.add(layers.Dense(40))
conv_base.trainable = True
model.summary()

os.makedirs("/media/dawars/hdd/dawars/Hand/temalab_ckpt/" + model_name, exist_ok=True)

checkpointer = ModelCheckpoint(
    "/media/dawars/hdd/dawars/Hand/temalab_ckpt/" + model_name + "/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5",
    monitor='val_loss', verbose=0,
    save_best_only=False, save_weights_only=True, mode='auto', period=3)
tensorboard = TensorBoard(log_dir='./logs/' + model_name)

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(),
              metrics=['acc'])

# serialize model to JSON
model_json = model.to_json()
with open(model_name + ".json", "w") as json_file:
    json_file.write(model_json)

# model.fit_generator(generate_batches(train_X, train_Y), steps_per_epoch=num_samples // 64)

model.fit(train_X, train_Y, batch_size=8, validation_split=.3, epochs=10000, shuffle=True,
          callbacks=[checkpointer, tensorboard])

# serialize weights to HDF5
model.save_weights(model_name + ".h5")
print("Saved model to disk, finished training")

import atexit


def exit_handler():
    print('Exiting app!')

    model.save_weights(model_name + "_exit.h5")
    print("Saved model to disk")


atexit.register(exit_handler)

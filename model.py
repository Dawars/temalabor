import keras

from load_hand_data import load_data, shuffle_data, preprocess_label, preprocess_feature


from keras.applications.mobilenet import MobileNet, decode_predictions
from keras import models
from keras import layers

model_name = "batch_32_epoch_50_data_10"

# loading datasets
train_X, train_Y = shuffle_data(*load_data(['Train1'], ['female'], read_labels=True))
train_X = preprocess_feature(train_X)
train_Y = preprocess_label(train_Y)

print("train x {} train y {}".format(train_X.shape, train_Y.shape))
# print("test x {} test y {}".format(len(test_X), len(test_Y)))
# print(train_X[:10])
# print(train_Y[:10])
# print("-----------------")
# print(test_X[:10])
# print(test_Y[:10])

# building model

conv_base = MobileNet(weights='imagenet',
                      input_shape=(224, 224, 3),
                      include_top=False)

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='selu'))
model.add(layers.Dense(40))
conv_base.trainable = False
# model.summary()

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

# model.fit_generator(generate_batches(train_X, train_Y), steps_per_epoch=num_samples // 64)


model.fit(train_X, train_Y, batch_size=32, validation_split=.3, epochs=50)

# serialize model to JSON
model_json = model.to_json()
with open(model_name + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_name + ".h5")
print("Saved model to disk")

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])

# TODO parse fx

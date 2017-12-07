import keras
from keras.applications import mobilenet
from keras.models import model_from_json

from load_hand_data import load_data, shuffle_data, preprocess_label, preprocess_feature

test_X, test_Y = load_data(['Test'], ['male'], read_labels=True)

test_X = preprocess_feature(test_X)
test_Y = preprocess_label(test_Y)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={
    'relu6': mobilenet.relu6,
    'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error',
                     optimizer=keras.optimizers.RMSprop(lr=2e-5),
                     metrics=['acc'])

score = loaded_model.evaluate(test_X, test_Y)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

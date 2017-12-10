import keras
from keras.applications import mobilenet
from keras.models import model_from_json

from load_hand_data import load_data, shuffle_data, preprocess_label, preprocess_feature

test_X, test_Y = load_data(['Test'], ['male'], read_labels=True)

test_X = preprocess_feature(test_X)
test_Y = preprocess_label(test_Y)

model_names = ['batch_32_epoch_50_data_10',
               'batch_32_epoch_200_data_30_adam',
               'batch_16_epoch_100_data_30_adam_0_1_random', ]

for model_name in model_names:
    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={
        'relu6': mobilenet.relu6,
        'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")

    # mobilenet
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='mean_squared_error',
                         optimizer=keras.optimizers.Adam(),
                         metrics=['acc'])

    score = loaded_model.evaluate(test_X, test_Y)
    print(model_name + " %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

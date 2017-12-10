import keras
from keras.applications import mobilenet
from keras.models import model_from_json
from keras.preprocessing.image import load_img

from load_hand_data import parse_label, preprocess_feature

# load json and create model
json_file = open('batch_16_epoch_100_data_30_adam_0_1_random.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={
    'relu6': mobilenet.relu6,
    'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
# load weights into new model
loaded_model.load_weights(
    "/media/dawars/hdd/dawars/Hand/temalab_ckpt/batch_16_epoch_100_data_30_adam_0_1_random/weights.100-0.00-0.23.hdf5")

loaded_model.compile(loss='mean_squared_error',
                     optimizer=keras.optimizers.Adam(),
                     metrics=['acc'])

# input to neural network
url = "/home/dawars/datasets/Hand/SyntheticHand/Test/male/1/Depth/0000663.png"
X = url
X = preprocess_feature([X])

print(X.shape)

preds = loaded_model.predict(X, batch_size=1)

print(parse_label(preds[0]))

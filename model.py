from load_hand_data import load_data, shuffle_data

from tensorflow import keras

# loading datasets
train_X, train_Y = shuffle_data(*load_data(['Train1'], ['female'], read_labels=True))
test_X, test_Y = shuffle_data(*load_data(['Test'], ['male'], read_labels=True))

print("train x {} train y {}".format(len(train_X), len(train_Y)))
print("test x {} test y {}".format(len(test_X), len(test_Y)))
print(train_X[:10])
print(train_Y[:10])
print("-----------------")
print(test_X[:10])
print(test_Y[:10])


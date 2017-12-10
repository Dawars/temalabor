import cv2
import keras
import numpy as np
import os
from keras.preprocessing import image
from matplotlib import pyplot as plt

dataset_location = '/home/dawars/datasets/Hand/SyntheticHand/'


# dataset_location = '/Users/dawars/projects/temalabor/SyntheticHand/'


def load_data(datasets, genders, read_labels=False):
    out_labels = []
    out_urls = []

    # load labels
    for dataset in datasets:
        for gender in genders:
            path = os.path.join(dataset_location, dataset, gender)
            for person in os.listdir(path):
                if int(person) > 50:
                    continue  # reduce data

                if read_labels:
                    with open(os.path.join(path, person, 'joints.txt')) as joint_file:
                        # read joints
                        joint_list = []
                        for line in joint_file:
                            split = line.split(' ')

                            if str(split[0]) == '-1':  # skip this
                                continue

                            joint_list.append(float(split[4]))
                            joint_list.append(float(split[5]))

                            if split[0] == '19':  # if last join in hand

                                out_labels.append(joint_list)  # 40 coords for 20 (x, y)
                                joint_list = []

                depth_path = os.path.join(path, person, 'Depth')
                for filename in sorted(os.listdir(depth_path)):
                    # read file names
                    out_urls.append(os.path.join(depth_path, filename))
    return out_urls, out_labels


def preprocess_feature(img_urls):
    """
    Loads and normalizes/standardises images
    :param img_urls:
    :return:
    """
    list_of_imgs = []
    for img in img_urls:
        if not img.endswith(".png"):
            continue

        a = cv2.imread(img)
        a = cv2.resize(a, (224, 224))
        # , target_size=(224, 224))  # PIL image
        # plt.imshow(a)
        # print(a.shape)
        if a is None:
            print("Unable to read image", img)
            continue
        # list_of_imgs.append(keras.preprocessing.image.img_to_array(a))  # convert to np array
        list_of_imgs.append(a)
    train_data = np.array(list_of_imgs, )
    return (train_data - 127) / 255


# joints
def preprocess_label(joints):
    for joint in joints:
        for i in range(0, len(joint), 2):
            joint[i] = (joint[i] / 512) - 0.5
            joint[i + 1] = (joint[i + 1] / 424) - 0.5
    return np.array(joints)


# joint output
def parse_label(features):
    out = []
    for i in range(0, len(features), 2):
        out.append((features[i] + 0.5) * 512)
        out.append((features[i + 1] + 0.5) * 424)
    return out


def shuffle_data(a, b):
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_image(url):
    x = image.load_img(url, target_size=(224, 224))
    x = np.expand_dims(x, axis=0)
    return x


def generate_batches(urls, labels):
    batch_size = 64
    i = 0
    batch_img = []
    batch_label = []
    while 1:
        urls, labels = shuffle_data(urls, labels)
        for i in range(len(urls)):
            url = urls[i]
            label = labels[i]
            img = load_image(url)
            i += 1

            batch_img.append(img)
            batch_label.append(label)
            if i >= batch_size:
                i = 0

                yield (batch_img, batch_label)

                batch_img = []
                batch_label = []

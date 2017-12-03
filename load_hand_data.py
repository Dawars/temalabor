import numpy as np
import os
dataset_location = '/home/dawars/datasets/Hand/SyntheticHand/'


def load_data(datasets, genders, read_labels=False):
    out_labels = []
    out_urls = []

    # load labels
    for dataset in datasets:
        for gender in genders:
            path = os.path.join(dataset_location, dataset, gender)
            for person in os.listdir(path):
                if int(person) > 5:
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


def shuffle_data(a, b):
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

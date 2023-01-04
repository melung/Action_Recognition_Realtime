import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz

training_subjects = [
   1,2,3,5,6,7,8
]

training_cameras = [0, 1]
max_body = 1
num_joint = 26
max_frame = 200
toolbar_width = 30


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            benchmark='ours',
            part='eval'):

    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])+1
        subject_id = int(
            filename[filename.find('S') + 1:filename.find('S') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'ours':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
            #issample = istraining
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        # print(sample_label)
        # print(s)
        # print(np.shape(data))
        # print(data.shape[1])
        # print(data[2,0:10,8,0])
        # print(i)
        fp[i, :, 0:data.shape[1], :, :] = data

    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='../Action_Data_master/data')
    parser.add_argument('--out_folder', default='data/ours_test')

    benchmark = ['ours']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                benchmark= b,
                part=p)

import numpy as np
import os


def read_skeleton(file):
    #print(file)
    #print(file)
    with open(file, 'r') as f:
        skeleton_sequence = {}
        mm = int(f.readlines()[-1])
        if mm >200:
            mm = 200
        skeleton_sequence['numFrame'] = mm
        skeleton_sequence['frameInfo'] = []
        f.close()

    with open(file, 'r') as f:
        f.readline()
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=1, num_joint=26):
    seq_info = read_skeleton(file)
    a = 0
    b = 0
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):

        if a==1:
            b = 1
            #print('nan')

            a = 0

        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    #print(v['x'])
                    #if np.isnan(v['x']) or np.isnan(v['y']) or np.isnan(v['z']):
                    if np.isnan(v['x']):
                        b = 1
                        a = 1
                        v['x'] = 0
                    if np.isnan(v['y']):
                        b = 1
                        v['y'] = 0
                    if np.isnan(v['z']):
                        b = 1
                        v['z'] = 0

                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
        if b == 1:
            print(file)

    return data
from mlsocket import MLSocket

import socket
import numpy as np
from Skeleton_Fusion import Skeleton_Fusion
import multiprocessing as mp
from multiprocessing import shared_memory
import timeit
import time
import os
import argparse
import sys
# torchlight
import torch
import torchlight
from torchlight import import_class
from processor import recognition
import keyboard
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#HOST = '169.254.164.143'#Master 컴퓨터 ip
HOST = '192.168.0.5'#Master 컴퓨터 ip

#CLIENTS_LIST = ['169.254.164.144', '169.254.164.145','169.254.164.146'] #slave 컴퓨터 IP EX) 3개 컴퓨터 이용시 [A, B, C]
CLIENTS_LIST = ['192.168.0.5', '169.254.164.145','169.254.164.146'] #slave 컴퓨터 IP EX) 3개 컴퓨터 이용시 [A, B, C]

Port = [1144, 1145, 1146] #slave 컴퓨터마다 통신할 port 지정 EX) 3개 컴퓨터 이용시 [가, 나, 다]
start_port = 12345
end_port = 4444
num_com = 1 #slave 컴퓨터개수
num_source = 2 #한 slave 컴퓨터에 연결된 kinect 개수
fps_cons = 20 #fps제한 (컴퓨터에 따라 다를 수 있습니다. 적절히 조절 slave 컴퓨터의 kinect fps와 같도록)
filtering_threshold = 0.7

Vive_ip = '192.168.0.9'
Vive_port = 1234


action_list = ["Hands up","Forward hand","T-pose","No action","Crouching","Open","Change grenade","Throw high","Throw low","Bend","Lean","Walking","Run","Shooting Pistol","Reload Pistol","Crouching Pistol","Change Pistol","Shooting Rifle","Reload Rifle","Crouching Rifle","Change Rifle","Holding high knife","Stabbing Knife","Change knife"]


vis = True




s_idx = '0 22 23 24 0 18 19 20 0 1 2 3 2 11 12 13 14 15 14 2 4 5 6 7 8 7'
f_idx = '22 23 24 25 18 19 20 21 1 2 3 26 11 12 13 14 15 16 17 4 5 6 7 8 9 10'
s_idx = np.asarray(s_idx.split(), dtype=int)
f_idx = np.asarray(f_idx.split(), dtype=int)
cur_time = str(time.localtime().tm_year)+str(time.localtime().tm_mon) +str(time.localtime().tm_mday)+str(time.localtime().tm_min)

# Press the green button in the gutter to run the script.
def get_data(HOST, PORT, CLIENTS_LIST, shm_nm, nptype, kk):
    temp_shm = shared_memory.SharedMemory(name = shm_nm)
    arr = np.frombuffer(buffer=temp_shm.buf, dtype=nptype)

    ADDR = (HOST, PORT + 100*kk)
    serverSocket = MLSocket()
    serverSocket.bind(ADDR)

    while True:

        serverSocket.listen(0)
        clientSocket, addr_info = serverSocket.accept()

        #print(addr_info[0])
        data = clientSocket.recv(1024)
        if addr_info[0] == CLIENTS_LIST[0]:
            arr[:] = data
        elif addr_info[0] == CLIENTS_LIST[1]:
            arr[:] = data

def get_vive_data(HOST, PORT, CLIENTS_LIST, shm_nm, nptype):
    temp_shm = shared_memory.SharedMemory(name = shm_nm)
    arr = np.frombuffer(buffer=temp_shm.buf, dtype=nptype)

    ADDR = (HOST, PORT)

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind(ADDR)

    while True:
        serverSocket.listen(0)
        clientSocket, addr_info = serverSocket.accept()

        #print(addr_info[0])
        data = clientSocket.recv(1024)
        text = data.decode('utf-8')
        joint_array = np.array(text.split())
        joint_array = joint_array.astype(np.float)
        #print(joint_array)


        arr[:] = joint_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processor collection')
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])
    arg = parser.parse_args()
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    model = p.start()


    #data = data.float().to(self.dev)



    input_data = np.zeros((1,3,100,26,1))

    total_joints = np.zeros((num_source*num_com * 27 * 4,), dtype=float)


    for ii in range(num_com):
        for kk in range(num_source):
            locals()[f"com{ii}_{kk}_joints"] = np.zeros((27 * 4,), dtype=float)

    for ii in range(num_com):
        for kk in range(num_source):
            locals()[f"shm{ii}_{kk}"] = shared_memory.SharedMemory(create=True, size=locals()[f"com{ii}_{kk}_joints"].nbytes)
            locals()[f"shared_joints{ii}_{kk}"] = np.frombuffer(locals()[f"shm{ii}_{kk}"].buf, locals()[f"com{ii}_{kk}_joints"].dtype)
            locals()[f"shared_joints{ii}_{kk}"][:] = locals()[f"com{ii}_{kk}_joints"]

    vive_joints = np.zeros((3 * 3,), dtype=float)
    shm_vive = shared_memory.SharedMemory(create=True, size=vive_joints.nbytes)
    vive_shared_joints = np.frombuffer(shm_vive.buf, vive_joints.dtype)
    vive_shared_joints [:] = vive_joints


    for ii in range(num_com):
        for kk in range(num_source):
            locals()[f"com{ii}_{kk}"] = mp.Process(target=get_data, args=[HOST, Port[ii], CLIENTS_LIST, locals()[f"shm{ii}_{kk}"].name,
                                                                     locals()[f"com{ii}_{kk}_joints"].dtype, kk])

    vive_com = mp.Process(target=get_vive_data, args=[HOST, Vive_port, CLIENTS_LIST, shm_vive.name,
                                                             vive_joints.dtype])


    for ii in range(num_com):
        for kk in range(num_source):
            locals()[f"com{ii}_{kk}"].daemon = True
            locals()[f"com{ii}_{kk}"].start()
    vive_com.daemon = True
    vive_com.start()



    b = input("Press Enter to Start")

    for ii in range(num_com):
        for kk in range(num_source):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((CLIENTS_LIST[ii], start_port + kk*100))
            s.sendto(str(b).encode(), (CLIENTS_LIST[ii], start_port + num_source*100))
            s.close()

    k = 0
    Fused_skel = []
    softmax = torch.nn.Softmax(dim=1)

    if vis:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-7, 7)
        ax.set_box_aspect((2, 2, 3))

    while True:
        if k == 0:
            time.sleep(0.05)
        start_t = timeit.default_timer()

        for ii in range(num_com):
            for kk in range(num_source):
                locals()[f"com{ii}_{kk}_joints"][:] = locals()[f"shared_joints{ii}_{kk}"]
                total_joints[27*4*(2*ii+kk):27*4*(2*ii+kk+1)] = locals()[f"com{ii}_{kk}_joints"][:]

        vive_joints[:] = vive_shared_joints
        #print(vive_joints)
        Fuse = Skeleton_Fusion(camnum=num_source*num_com, joint=total_joints, pre_joint= [] if k == 0 else Fused_skel, vive_joints= vive_joints)
        Fused_skel = Fuse.Fusion()
        p = np.asarray(Fused_skel.vJointPositions)
        #print(p)
        np.nan_to_num(p, copy=0)
        # np.where(p == np.NaN, 0, p)
        #print(p)
        data = p[1:27,:].T
        #print(data)
        #print(np.shape(p))
        input_data[0,:,0:99,:,0] = input_data[0,:,1:100,:,0]
        input_data[0,:, 99, :, 0] = data

        if k > 150:
            data = torch.Tensor(input_data).float().to("cuda:0")
            with torch.no_grad():
                output = model(data)
                output = softmax(output)
                output = output.cpu().detach().numpy()
                #print(k)
                if np.isnan(np.max(output)):
                    print('Action : No action')
                    print('Confidence: nan')
                elif np.max(output) > filtering_threshold:
                    print('Action : ' + action_list[np.argmax(output)])
                    print('Confidence: ',np.round(np.max(output),2))

                else:
                    print('Action : No action')
                    print('Confidence: ',np.round(np.max(output),2))


        if vis:
            if k > 150:
                if np.isnan(np.max(output)):
                    plt.title('Action: No action\nConfidence: nan\nFrame: '+ str(k),loc='left')
                elif np.max(output) > filtering_threshold:
                    plt.title('Action: '+action_list[np.argmax(output)] +'\nConfidence: ' + str(np.round(np.max(output),3))+'\nFrame: '+ str(k),loc='left')
                else:
                    plt.title('Action: No action\nConfidence: ' + str(np.round(np.max(output),3))+'\nFrame: '+ str(k),loc='left')
            else:
                plt.title('Prepare the Action Recognition\nFrame: '+ str(k), loc='left')

            points = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='b')
            plt.pause(0.001)
            plt.draw()
            points.remove()

        k += 1
        terminate_t = timeit.default_timer()
        delay_t = 1/(fps_cons+10) - (terminate_t - start_t)
        if delay_t > 0:
            time.sleep(delay_t)
        terminate_t = timeit.default_timer()
        FPS = 1 / (terminate_t - start_t)

        print("FPS: " + str(FPS))
        #if k >300:
        if keyboard.is_pressed("q"):
            for ii in range(num_com):
                for kk in range(num_source):
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((CLIENTS_LIST[ii], end_port+100*kk))
                    s.close()
            break



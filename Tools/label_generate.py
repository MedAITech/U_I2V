#label文件的生成。生成的label文件格式同ucf101_train_split_1_rawframes.txt。
#原理：读取的数据

import cv2
import time
import os
# -*- coding:utf-8 -*-

input_folder = '/media/user/Disk1/chentingxiu/I2V/public/BUSV/malignant'  # video路径
input_files = os.listdir(input_folder)  #获取路径内所有video

file_path = '/media/user/Disk1/chentingxiu/I2V/public/BUSV/busv_label.txt'  #待写入的空文件
file = open(file_path, 'r+',encoding='utf-8',errors='ignore')
line = file.readlines()

n = 1
new_lines1 = []
new_lines2 = []
new_lines3 = []
for input_file in input_files:

    #写帧数
    video_path = os.listdir(os.path.join(input_folder, input_file))
    #video = cv2.VideoCapture(input_file)
    #frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = len(video_path)
    # label
    line = input_file
    # line = input_file.strip()  # 去除换行符
    line += ' ' + str(frame_count)
    #良恶性标签判断
    '''
    m = (line.split('('))[2]
    if m == 'begin':
        line += ' 0\n'  # 添加标签
        new_lines1.append(line)
    if m == 'malignant':
        line += ' 1\n'
        new_lines2.append(line)
    if m == 'normal':
        line += ' 2\n'
        new_lines2.append(line)
    '''
    line += ' 1\n'
    new_lines1.append(line)
    print("count:",n)
    n += 1

#frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
file.seek(0)  # 将文件指针移动到文件开头
file.writelines('M:')
file.writelines(new_lines1)  # 写入修改后的数据
print("label done!!!")
#file.writelines(new_lines2)
#print("label2 done!!!")
#file.writelines(new_lines3)
#print("normal done!!!")
file.close()  # 关闭文件
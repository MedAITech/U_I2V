#

import os
import subprocess
import shutil

list_file = '/home/ai1011/mmaction-master/data/ucf101/annotations/train_val/trainlist03.txt'
output_folder = '/media/user/Disk1/chentingxiu/C3D/03/videos/train'
initial_folder = '/home/ai1011/mmaction-master/data/ucf101/videos'
# 获取所有MP4文件的路径
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    x = t[0].strip().split('/')
    #y = t[0].strip().split('.')
    out_file = os.path.join(output_folder, x[0])
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    shutil.copy(initial_file, out_file)

initial_folder = '/home/ai1011/mmaction-master/data/ucf101/videos'
list_file = '/home/ai1011/mmaction-master/data/ucf101/annotations/train_val/testlist03.txt'
output_folder = '/media/user/Disk1/chentingxiu/C3D/03/videos/val'
# 获取所有MP4文件的路径
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    x = t[0].strip().split('/')
    #y = t[0].strip().split('.')
    out_file = os.path.join(output_folder, x[0])
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    shutil.copy(initial_file, out_file)

list_file = '/home/ai1011/mmaction-master/data/ucf101/annotations/train_test/testlist03.txt'
output_folder = '/media/user/Disk1/chentingxiu/C3D/03/videos/test'
# 获取所有MP4文件的路径
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    x = t[0].strip().split('/')
    #y = t[0].strip().split('.')
    out_file = os.path.join(output_folder, x[0])
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    shutil.copy(initial_file, out_file)
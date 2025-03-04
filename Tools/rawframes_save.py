import os
import subprocess
import shutil

#旧的C3D代码用到。按照train/test/eval形式存放视频帧。

def copy_file(src_path, dst_path):

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    else:
        print('this video is already here!!!!')
        return

    for file in os.listdir(src_path):
        src_file = os.path.join(src_path, file)
        dst_file = os.path.join(dst_path, file)

        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)
    return

initial_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/video/rawframes'

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/train/ucf101_train_split_1_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/03/rawframes/train'
# 获取所有MP4文件的路径
n=1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:',out_file)
    n+=1

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/train/ucf101_val_split_1_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/03/rawframes/val'
n=1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:', out_file)
    n += 1

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/test/ucf101_val_split_1_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/03/rawframes/test'

n = 1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:', out_file)
    n += 1
print('01 done!!!!')

# 设置输入和输出文件夹路径
initial_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/video/rawframes'

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/train/ucf101_train_split_2_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/02/rawframes/train'
# 获取所有MP4文件的路径
n=1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:',out_file)
    n+=1

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/train/ucf101_val_split_2_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/02/rawframes/val'
n=1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:', out_file)
    n += 1

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/test/ucf101_val_split_2_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/02/rawframes/test'

n = 1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:', n)
    n += 1
print('02 done!!!!')

initial_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/video/rawframes'

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/train/ucf101_train_split_3_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/03/rawframes/train'
# 获取所有MP4文件的路径
n=1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:',out_file)
    n+=1

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/train/ucf101_val_split_3_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/03/rawframes/val'
n=1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:', out_file)
    n += 1

list_file = '/media/user/Disk1/chentingxiu/I2V/乳腺/test/ucf101_val_split_3_rawframes.txt'
output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/C3D/03/rawframes/test'

n = 1
tmp = [x.strip().split(' ') for x in open(list_file)]
for t in tmp:
    initial_file = os.path.join(initial_folder, t[0])
    out_file = os.path.join(output_folder, t[0])
    copy_file(initial_file, out_file)
    print('done:', out_file)
    n += 1
print('03 done!!!!')
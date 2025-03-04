import os
import shutil

video_folders = '/media/user/Disk1/chentingxiu/I2V/public/split/'
videos = os.listdir(video_folders)
for video in videos:
    initial_path = video_folders + video + '/000000.png'
    aim_path = '/media/user/Disk1/chentingxiu/I2V/public/BUSV/initial_frames/' + video + '.png'
    # 复制文件
    shutil.copy(initial_path, aim_path)

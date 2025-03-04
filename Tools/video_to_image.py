# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/2/20  9:07
# @Author  : Cao Xu
# @FileName: video_to_image.py
"""
Description:   video to images
"""
import os
import cv2

if __name__ == '__main__':
    x = 0
    video_files = [x for x in os.listdir('./') if x.endswith(('.mp4', '.mkv', '.avi', '.wmv', '.mov', '.flv'))]
    for f in video_files:
        print('deal with :',x,f)
        out_path = f.split('.')[0]
        #os.makedirs(out_path, exist_ok=True)
        video = cv2.VideoCapture('./' + f)
        frame_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
        # print(f, frame_len)
        for i in range(frame_len - 1):
            # try:
            ret, frame = video.read()
            if frame is not None:
                #import pdb;pdb.set_trace()
                save_path = os.path.join('./rawframes/'+ out_path + '/'+ str(i) + '.png')
                test_path = os.path.join('./rawframes/'+ out_path)
                if not os.path.exists(test_path):
                    os.makedirs(test_path)
                cv2.imwrite(save_path, frame)
        print('deal done!!!',out_path)
        x += 1
        #import pdb;pdb.set_trace()

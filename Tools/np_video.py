# -*- coding: utf-8 -*-

import numpy as np
import cv2

def video_to_numpy_array(video_path):

    rgb_frame_list = []
    video_read_capture = cv2.VideoCapture(video_path)
    while video_read_capture.isOpened():
        result, frame = video_read_capture.read()
        if not result:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame_list.append(rgb_frame)

    video_read_capture.release()

    video_nparray = np.array(rgb_frame_list)

    return video_nparray

def numpy_array_to_video(numpy_array,video_out_path):
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]

    out_video_size = (video_width,video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, 30, out_video_size)

    for frame in numpy_array:
        video_write_capture.write(frame)

    video_write_capture.release()


if __name__ == '__main__':
    video_input_path = r'/media/user/Disk1/chentingxiu/I2V/乳腺/video/shyy_ID-00000001_label1_M201901221831582180015.avi'
    video_output_path = r'/media/user/Disk1/chentingxiu/I2V/乳腺/example.mp4'

    # 将视频读取为维度为N * W * H * C的numpy数组
    video_np_array = video_to_numpy_array(video_input_path)

    # 将numpy数组写为视频
    numpy_array_to_video(video_np_array,video_output_path)
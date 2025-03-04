import cv2
import time
import os
import subprocess

def video_cut(input_path,output_path):

    #.time()方法用来记录当前时间
    start_time = time.time()

    # 导入一个本地视频赋给cap
    cap = cv2.VideoCapture(input_path)

    # 获取视频的帧速率fps
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 获取视频的宽高
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    #设置视频的解码器
    fourcc = cv2.VideoWriter_fourcc('D','I','V','X')

    # 获取视频的总帧数
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 5秒的总帧率
    framss = 0.5 * fps
    print('视频详细信息：\n\tfps：',fps,'\n\t宽、高：',size,'\n\t总帧数：',frames,'\n\t解码器：',fourcc)

    # i用来计数每一帧视频，n用来计数视频个数
    i = 0
    n = 1

    #获取文件名，方便后面给分割出的视频片段加编号,这部分需要自己看一下，按照自己期望生成的文件名格式修改
    out_video = input_path.split("/")
    video_name = out_video[2]
    name = (video_name.split('.'))[0]

    # 创建一个视频对象，并依次设置输出路径下的名称、解码器、每秒帧率、视频宽高
    print('写入第%d个视频中...' % n)
    #out = cv2.VideoWriter('/home/ai1011/mmaction-master/data/ucf101/video_seg/Benign/HX-00001259_1_良_'+str(n)+'.avi',fourcc,fps,size)
    out = cv2.VideoWriter(output_path + name + '('+str(n) + ').avi', fourcc, fps, size)

    while True:
        # .read()方法将会返回一个布尔值和一组矩阵
        ret,frame = cap.read()
        if ret:
            out.write(frame)
            i = i + 1
            if(i % framss == 0):
                out.release()
                n = n + 1
                print('写入第%d个视频中...'%n)
                out = cv2.VideoWriter(output_path + name + '('+str(n) + ').avi', fourcc, fps, size)

        else:
            final_time = time.time() - start_time
            print('...结束，本次处理输出共',n,'个视频,耗时约%0.2fs'%final_time)
            break

    # 释放内存资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return True

#input_folder = './Benign'
#output_folder = './Benign_seg/'
#input_folder = './Malignant'
#output_folder = './Malignant_seg/'
input_folder = './Normal'
output_folder = './Normal_seg/'

# 获取所有MP4/avi文件的路径
input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.avi')]
flag = 0

# 循环处理每个输入文件
for input_file in input_files:
    # 构建输出文件路径
    flag += 1
    x = video_cut(input_file, output_folder)
    print("&d video was done!", flag)

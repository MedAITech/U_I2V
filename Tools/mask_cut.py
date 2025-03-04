import cv2
import time
import os
import subprocess
import pandas as pd
import random
from PIL import Image
import re

def resize(img,target_size):

    old_size= img.shape[0:2]
    ratio = min(float(target_size[i])/(max(old_size[i],1)) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio + 0.5) for i in old_size])

    img = cv2.resize(img,(new_size[1], new_size[0]),interpolation=cv2.INTER_LINEAR)
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]

    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
    return img_new

def cut_and_resize(img,size,mask):
    mask_re = re.findall('\d+', mask[0])
    mask = [int(val) for val in mask_re]
    x1 = (size[0] - mask[0]) // 2
    y1 = (size[1] - mask[1]) // 2
    x2 = x1+mask[1]
    y2 = y1+mask[0]

    #numpy数组裁剪
    #print(x1,x2,y1,y2,img.shape,mask)
    img1 = img[x1:x2,y1:y2,:]
    #img1 = img1[:,y1:y2,:]
    img_new = resize(img1,[224,224])
    return img_new

def mask_cut(input_path,output_path):
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

    # 截3个视频
    framss = frames // 3
    print('视频详细信息：\n\tfps：',fps,'\n\t宽、高：',size,'\n\t总帧数：',frames,'\n\t解码器：',fourcc,'\n\t每段视频帧数：',framss)

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
    out = cv2.VideoWriter(output_path + name + '('+str(n) + ').avi', fourcc, fps, (224,224))

    #每个视频选择三个mask尺寸
    mask_count = 0
    mask = random.sample(range(0, (len(mask_list)) - 1), 3)
    while True:
        # .read()方法将会返回一个布尔值和一组矩阵
        ret,frame = cap.read()
        #print(frame.shape)
        if ret:
            choose = mask[mask_count]
            mask_now = mask_list[choose]
            new_frame = cut_and_resize(frame, size, mask_now)
            out.write(new_frame)
            i = i + 1
            if((i % framss == 0) and (mask_count < 2)):
                print("mask 尺寸：")
                print(mask_now)
                mask_count += 1
                out.release()
                n = n + 1
                print('写入第%d个视频中...'%n)
                out = cv2.VideoWriter(output_path + name + '('+str(n) + ').avi', fourcc, fps, (224,224))

        else:
            final_time = time.time() - start_time
            print('...结束，本次处理输出共',n,'个视频,耗时约%0.2fs'%final_time)
            break

    # 释放内存资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# 读取Excel文件
sheet = pd.read_excel('稳态视频统计检测框数量.xlsx',sheet_name = 'Sheet2')

# 将表格转换为列表
mask_list = sheet.values.tolist()

#input_folder = '/home/ai1011/mmaction-master/data/ucf101_all/videos/Normal'
#output_folder = './Normal_mask/'

#input_folder = '/home/ai1011/mmaction-master/data/ucf101_all/videos/Benign'
input_folder = './Normal'
output_folder = './Normal_mask/'

# 获取所有MP4/avi文件的路径
input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.avi')]
flag = 0
# 循环处理每个输入文件
for input_file in input_files:
    # 构建输出文件路径
    flag += 1
    x = mask_cut(input_file, output_folder)
    print("&d video was done!", flag)
    #break


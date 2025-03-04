
import os

# 获取指定路径的所有文件夹和文件名称
dir_list = os.listdir('D:/CTX/赵医生科研项目/秀秀label')
# 筛选文件夹名称并保存到列表中
folders = []

#类别计数器
gray = 0
video_gray = 0
blood = 0
video_blood = 0
file_count = 0

#遍历文件夹并计数
for folder in dir_list:

    file_count += 1
    print("正在处理的文件夹:",folder)

    labels = os.listdir('D:/CTX/赵医生科研项目/秀秀label/' + folder)
    for label in labels:
        if label == "gray":
            gray_files = os.listdir('D:/CTX/赵医生科研项目/秀秀label/' + folder +'/'+ label)
            gray_num = len(gray_files)
            gray = gray_num + gray
            print("gray:%d"%gray_num)
        if label == "video_gray":
            video_gray_files = os.listdir('D:/CTX/赵医生科研项目/秀秀label/' + folder + '/' + label)
            video_gray_num = len(video_gray_files)
            video_gray = video_gray_num + video_gray
            print("video_gray:%d" % video_gray_num)
        if label == "blood":
            blood_files = os.listdir('D:/CTX/赵医生科研项目/秀秀label/' + folder + '/' + label)
            blood_num = len(blood_files)
            blood = blood_num + blood
            print("blood:%d" % blood_num)
        if label == "video_blood":
            video_blood_files = os.listdir('D:/CTX/赵医生科研项目/秀秀label/' + folder + '/' + label)
            video_blood_num = len(video_blood_files)
            video_blood = video_blood_num + video_blood
            print("video_blood:%d" % video_blood_num)

print("total gray:%d"%gray)
print("total video_gray:%d"%video_gray)
print("total blood:%d"%blood)
print("total video_blood:%d"%video_blood)
print("total files:%d"%file_count)

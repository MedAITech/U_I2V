#处理产品线乳腺数据用到。

import os
import shutil

initial_folder = 'D:/CTX/Data/I2V/视频转图片已完成/颐和已完成/06薛林_L中：99%恶(病理)/list.txt'
initial = 'D:/CTX/Data/I2V/视频转图片已完成/颐和已完成/06薛林_L中：99%恶(病理)/结节/'
out_folder = 'C:/临时存放文件/I2V_data/14/2/'
n = 1
m = str(n)
out_file = os.path.join(out_folder, m)
if not os.path.exists(out_file):
    os.makedirs(out_file)
    print(out_file)

tmp = [x.strip().split(' ') for x in open(initial_folder)]

i = 0
for t in tmp:
    x = t[0].strip().split('.')
    y=int(x[0])
    #首帧不需要pre,直接保存到当前文件夹
    if i == 0:
        print('save first frame')
        pre = y
    if i > 0:
        if (y == pre + 1):
           print('next:',t)
        else:
            n += 1
            m = str(n)
            out_file = os.path.join(out_folder, m)
            if not os.path.exists(out_file):
                os.makedirs(out_file)
                print(out_file)
        pre = y
    initial_file = os.path.join(initial, t[0])
    output_file = os.path.join(out_file,'/',t[0])
    shutil.copy(initial_file, out_file)
    i += 1
    #print(y)
print('done!')
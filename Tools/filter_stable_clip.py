import torch
import cv2
import  numpy as np
import os,json
import time,warnings
from torchvision import transforms
from tqdm import tqdm
import math
import copy
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda"
class_map = {0:"benign", 1:"malig", 2:"normal"}
class_color = {0:(0, 255, 0), 1:(0, 0, 255), 2:(255, 255, 255)}

mean = [0.485, 0.456, 0.406]
std = [0.228, 0.224, 0.225]
transformations = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])


def mask2bbox(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x,y,w,h])
    if len(contours) == 0:
        boxes = None
    return contours, boxes

def crop_roi_extendx(box,img,ratio=0.25):
    size = img.shape
    max_length = max(box[2], box[3])
    ex_pixel = int(max_length * ratio)
    dim1_cut_min = box[1] - ex_pixel
    dim1_cut_max = box[1] + box[3] + ex_pixel
    dim2_cut_min = box[0] - ex_pixel
    dim2_cut_max = box[0] + box[2] + ex_pixel

    if dim1_cut_min < 0:
        dim1_cut_min = 0
    if dim2_cut_min < 0:
        dim2_cut_min = 0
    if dim1_cut_max > size[0]:
        dim1_cut_max = size[0] - 1
    if dim2_cut_max > size[1]:
        dim2_cut_max = size[1] - 1
    roi = [dim1_cut_min, dim1_cut_max+1, dim2_cut_min, dim2_cut_max+1]

    return img[dim1_cut_min:dim1_cut_max+1, dim2_cut_min:dim2_cut_max+1], roi

def resize_img_keep_ratio(img,target_size=[224,224]):

    old_size= img.shape[0:2]
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio + 0.5) for i in old_size])

    img = cv2.resize(img,(new_size[1], new_size[0]),interpolation=cv2.INTER_LINEAR)
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]

    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
    return img_new

def seg_preprocess(img,size):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != [size[0], size[1]]:
        image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
    image = image / 255.0
    image = image.transpose(2, 0, 1).astype('float32')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = DEVICE
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    return  x_tensor

def cls_preprocess(image,size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (size[0], size[1]):
        image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
    image = image / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    input = torch.FloatTensor(image[np.newaxis, :, :])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = DEVICE
    x_tensor = input.to(device)
    return  x_tensor

def compare_mask_intersection(nodu_mask, gland_mask, threshold):
    # 将掩码图像转换为二值图像
    _, nodu_mask = cv2.threshold(nodu_mask, 1, 255, cv2.THRESH_BINARY)
    _, gland_mask = cv2.threshold(gland_mask, 1, 255, cv2.THRESH_BINARY)

    # 计算两个掩码图像的交集
    intersection = cv2.bitwise_and(nodu_mask, gland_mask)
    # 计算交集的面积
    intersection_area = np.count_nonzero(intersection)
    # 计算两个掩码图像的面积
    area = np.count_nonzero(nodu_mask)

    if area==0:
        return False
    # 计算交集面积占两个图像面积的比例
    intersection_ratio = intersection_area / area
    # 判断交集面积比例是否小于阈值
    if intersection_ratio > threshold:
        return True
    else:
        return False

def check_box_shift(frame_que, box2check):
    def calculate_center_distance(box1, box2):
        # box1 和 box2 分别是两个矩形框的边界框：(x, y, w, h)
        # 计算第一个矩形框的中心点坐标
        box1_center_x = box1[0] + box1[2] // 2
        box1_center_y = box1[1] + box1[3] // 2
        # 计算第二个矩形框的中心点坐标
        box2_center_x = box2[0] + box2[2] // 2
        box2_center_y = box2[1] + box2[3] // 2
        # 计算中心点之间的距离
        distance = math.sqrt((box2_center_x - box1_center_x)**2 + (box2_center_y - box1_center_y)**2)
        return distance
    distance_thres = 50
    match_num = 0
    frame_num_with_box = 0
    for frame in frame_que:
        if frame:
            for box in frame:
                if calculate_center_distance(box, box2check) < distance_thres:
                    match_num+=1
                    break
            frame_num_with_box+=1
    if match_num>=15:
        return True
    else:
        return False


def cal_box_shift(box1, box2):
    # box1 和 box2 分别是两个矩形框的边界框：(x, y, w, h)
    # 计算第一个矩形框的中心点坐标
    box1_center_x = box1[0] + box1[2] // 2
    box1_center_y = box1[1] + box1[3] // 2
    # 计算第二个矩形框的中心点坐标
    box2_center_x = box2[0] + box2[2] // 2
    box2_center_y = box2[1] + box2[3] // 2
    # 计算中心点之间的距离
    if abs(box2_center_x - box1_center_x)<40 and abs(box2_center_y - box1_center_y)<40:
        return True
    else:
        return False

def seg_inference(task, seg_model, input_tensor, img_size):
    with torch.no_grad():
        mask = seg_model(input_tensor)[0]
        mask = mask.squeeze().cpu().numpy().round()
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 255
        mask = mask.astype(np.uint8)

        if mask.shape != img_size:
            mask = cv2.resize(mask, (img_size[1], img_size[0]), cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        if task == "nod_seg":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=3)
            mask = opened
            nod_contours, boxes = mask2bbox(mask)
            return mask, nod_contours, boxes
        elif task == "gland_seg":
            gland_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            return mask, gland_contours
        else:
            print("unexpected task!")

def cls_inference(cls_model, input_tensor):
    with torch.no_grad():
        pred = cls_model(input_tensor)
        pred = torch.softmax(pred,dim=1)
        pred_prob, pred_label = torch.max(pred, 1)
        pred_label = pred_label.cpu().numpy()[0]
        pred_prob = pred_prob.cpu().numpy()[0]
        return pred_label, pred_prob


def cal_box_iou(box1, box2):
    x1_1, y1_1, w1_1, h1_1 = box1
    x1_2, y1_2, w1_2, h1_2 = box2
    
    area1 = w1_1 * h1_1
    area2 = w1_2 * h1_2
    
    intersection_x1 = max(x1_1, x1_2)
    intersection_y1 = max(y1_1, y1_2)
    intersection_x2 = min(x1_1 + w1_1, x1_2 + w1_2)
    intersection_y2 = min(y1_1 + h1_1, y1_2 + h1_2)
    
    if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
        intersection_area = 0
    else:
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    
    iou = intersection_area / (area1 + area2 - intersection_area)
    
    return iou 

def save_video(video_path, frames, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编解码器
    out = cv2.VideoWriter(video_path, fourcc, fps, size)
    for frame in frames:
        out.write(frame)
    out.release()
    cv2.destroyAllWindows() 



# 结节区域外扩
def crop_roi_extend(box, img, ratio=0.25, size=[]):
    x, y, w, h = box
    max_length = max(w, h)
    expand_pixel = int(max_length * ratio)

    dim1_cut_min = max(0, y - expand_pixel)
    dim1_cut_max = min(size[0] - 1, y + h + expand_pixel)
    dim2_cut_min = max(0, x - expand_pixel)
    dim2_cut_max = min(size[1] - 1, x + w + expand_pixel)
   
    new_box = [dim2_cut_min, dim1_cut_min, dim2_cut_max-dim2_cut_min, dim1_cut_max-dim1_cut_min]
    return img[dim1_cut_min:dim1_cut_max + 1, dim2_cut_min:dim2_cut_max + 1], new_box


# 保持长宽比缩放图像
def resize_img_keep_ratio(img, target_size=[224,224]):
    old_height, old_width = img.shape[:2]
    ratio = min(target_size[0] / old_height, target_size[1] / old_width)
    new_height = int(old_height * ratio + 0.5)
    new_width = int(old_width * ratio + 0.5)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    pad_height = target_size[0] - new_height
    pad_width = target_size[1] - new_width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    img_new = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=[0, 0, 0])
    return img_new


def filter_stable_videos(ext_ratio, 
                         video_path, 
                         output_path_complete,
                         output_path_crop,
                         output_path_mask, 
                         video_gt, 
                         glandseg_model, 
                         nodseg_model):
    frame_num_box_more_than_one =0
    valid_frame_num = 0
    video_name = video_path.split('/')[-1][:-4]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编解码器
    out_video_complete = cv2.VideoWriter(output_path_complete, fourcc, fps, size)
    out_video_crop = cv2.VideoWriter(output_path_crop, fourcc, fps, (224, 224))
    out_video_mask = cv2.VideoWriter(output_path_mask, fourcc, fps, size)

    if cap.isOpened():
        rval,frame = cap.read()
        print("open")
    else:
        print("false")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    box_w_list=[]
    box_h_list=[]
    with tqdm(total=total_frames) as pbar:
        while rval:
            rval,frame = cap.read()
            pbar.update(1)
            if not rval:
                break
            ori_image = copy.deepcopy(frame)
            img_size = ori_image.shape
            img_tensor = seg_preprocess(ori_image, (512, 512))
            gland_mask, gland_contours = seg_inference("gland_seg", glandseg_model, img_tensor, img_size)
            nod_masks, nod_contours, boxes = seg_inference("nod_seg", nodseg_model, img_tensor, img_size)
            # cv2.drawContours(frame, gland_contours, -1, (255, 255, 0), 2) #绘制腺体轮廓

            clean_mask_boxes = []
            for cont_id, contour in enumerate(nod_contours):
                one_nod_mask = np.zeros_like(nod_masks)
                cv2.drawContours(one_nod_mask, [contour], 0, 255, -1)
                box_w, box_h = boxes[cont_id][2], boxes[cont_id][3]
                # 用不同颜色分别绘制和腺体有/无交集的结节
                if compare_mask_intersection(one_nod_mask, gland_mask, 0) and max(box_w, box_h) * 0.0062 > 0.186:
                    # cv2.drawContours(frame, [contour], -1, (0, 165, 255), 2)
                    clean_mask_boxes.append(boxes[cont_id])

            if not clean_mask_boxes:
                continue
            else:
                if len(clean_mask_boxes)>1:
                    frame_num_box_more_than_one+=1
                else:
                    valid_frame_num+=1
                    det_img, new_box = crop_roi_extend(clean_mask_boxes[0], ori_image, ext_ratio, img_size)
                    resized_img = resize_img_keep_ratio(det_img)
                    out_video_crop.write(resized_img)

                    x,y,w,h = new_box
                    draw_mask = np.zeros(img_size, dtype=np.uint8)
                    cv2.rectangle(draw_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    box_w_list.append(clean_mask_boxes[0][2])
                    box_h_list.append(clean_mask_boxes[0][3])

                    out_video_complete.write(frame)
                    out_video_mask.write(draw_mask)
  
    cap.release()
    cv2.destroyAllWindows()
    out_video_crop.release()
    out_video_complete.release()
    out_video_mask.release()
    
    if box_w_list and box_h_list:
        box_w_avg = int(sum(box_w_list)/len(box_w_list))
        box_h_avg = int(sum(box_h_list)/len(box_h_list))
    else:
        box_w_avg=0
        box_h_avg=0
    return total_frames, frame_num_box_more_than_one, valid_frame_num, box_w_avg, box_h_avg


def get_frame_num(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames


thyroidSeg_weight_path = "/media/user/Disk1/yanbingcong/models/thyroid/thyroidseg_model/best_0.8346.pth"
thyroidSeg_model = torch.load(thyroidSeg_weight_path,map_location=DEVICE)
thyroidSeg_model.eval()

nodSeg_weight_path = "models/ini_models/thyroid_noduleSeg_Uneteb4_ep207_0.8213.pth"
nodSeg_model = torch.load(nodSeg_weight_path,map_location=DEVICE)
nodSeg_model.eval()

# nodClassify_weight_path = "models/ini_models/thyroid_nodClassify_densenet161_dynamic_crop_14aug_20230707.pt"
# nodClassify_model = torch.load(nodClassify_weight_path,map_location=DEVICE)
# nodClassify_model.eval()


video_dir = "/media/user/Disk1/yanbingcong/data/ThyroidData/thyroid_video/视频片段_明确病理_裁剪后/汇总_裁剪稳态片段"
out_dir = "/media/user/Disk1/yanbingcong/data/ThyroidData/thyroid_video/视频片段_明确病理_裁剪后/汇总_裁剪稳态片段_统计结果"
video_out_dir_complete = "/media/user/Disk1/yanbingcong/data/ThyroidData/thyroid_video/视频片段_明确病理_裁剪后/汇总_裁剪稳态片段_筛选单检测框/整图"
video_out_dir_crop = "/media/user/Disk1/yanbingcong/data/ThyroidData/thyroid_video/视频片段_明确病理_裁剪后/汇总_裁剪稳态片段_筛选单检测框/patch"
video_out_dir_mask = "/media/user/Disk1/yanbingcong/data/ThyroidData/thyroid_video/视频片段_明确病理_裁剪后/汇总_裁剪稳态片段_筛选单检测框/整图mask"

os.makedirs(video_out_dir_complete, exist_ok=True)
os.makedirs(video_out_dir_crop, exist_ok=True)
os.makedirs(video_out_dir_mask, exist_ok=True)

ext_ratio=0.25
label_map={"良":0, "恶":1}

df = pd.read_excel("/media/user/Disk1/yanbingcong/data/ThyroidData/thyroid_video/视频片段_明确病理_裁剪后/汇总_裁剪稳态片段_统计结果/acc_时长_统计.xlsx")
valid_clips = []
for i in range(len(df)):
    if int(df.iloc[i]["时长(帧)"])>=15:
        valid_clips.append(df.iloc[i]["视频"])
print(f"稳态时长超过15帧的数量：{len(valid_clips)}")
idx = 0
vid_list = []
total_frame_num_list = []
frame_num_box_more_than_one_list = []
valid_frame_num_list = []
box_w_h_list = []

for video_name in os.listdir(video_dir):
    if video_name not in valid_clips:
        continue
    output_path_complete = os.path.join(video_out_dir_complete, video_name)
    output_path_crop = os.path.join(video_out_dir_crop, video_name)
    output_path_mask = os.path.join(video_out_dir_mask, video_name)

    print(f"第{idx+1}个视频")
    idx+=1
    video_path = os.path.join(video_dir, video_name)
    video_gt = label_map[video_name.split('_')[-2]]
   
    total_frames, frame_num_box_more_than_one, valid_frame_num, box_w, box_h = filter_stable_videos(ext_ratio, 
                                                    video_path,
                                                    output_path_complete,
                                                    output_path_crop,
                                                    output_path_mask,
                                                    video_gt, 
                                                    glandseg_model=thyroidSeg_model, 
                                                    nodseg_model=nodSeg_model, 
                                                )
   
    vid_list.append(video_name)
    total_frame_num_list.append(total_frames)
    frame_num_box_more_than_one_list.append(frame_num_box_more_than_one)
    valid_frame_num_list.append(valid_frame_num)
    box_w_h_list.append((box_w, box_h))
    
    # if idx>5:
    #     break

df_res = pd.DataFrame({"视频":vid_list,
                       "帧数":total_frame_num_list,
                       "检测框大于1的帧数":frame_num_box_more_than_one_list,
                       "检测框等于1的帧数":valid_frame_num_list,
                       "检测框大小":box_w_h_list,
                       })
df_res.to_excel(os.path.join(out_dir, "稳态视频统计检测框数量temp.xlsx"), index=False)


import os
'''
folder_path = "your_folder_path"
if not os.listdir(folder_path):
    print("该文件夹为空")
'''
try:

    folder_path = "./Malignant"
    for sub_folder_name in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        if os.path.isdir(sub_folder_path):

            if not os.listdir(sub_folder_path):
                os.rmdir(sub_folder_path)
                print(f"{sub_folder_name} 已经删除")
            else:
                print(f"{sub_folder_name} 文件夹非空")
        else:
            print(f'{sub_folder_name}不是文件夹')
            continue
except PermissionError as e:
    print(f'拒绝访问')
except FileNotFoundError as e:
    print('路径错误，请检查路径')
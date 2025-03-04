import os
import subprocess
'''
#一、mp42avi,ffmpeg方法
# 设置输入和输出文件夹路径
input_folder = './Normal'
output_folder = './output'

# 获取所有MP4文件的路径
input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.mp4')]

# 循环处理每个输入文件
for input_file in input_files:
    # 构建输出文件路径
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + '.avi')

    # 调用FFmpeg命令进行转换
    subprocess.call(['ffmpeg', '-i', input_file, output_file])
'''

#二、wmv2mp4，moviepy方法
from moviepy.editor import VideoFileClip
import os
def convert_to_mp4(input_path, output_path):
    try:
        # Load the video clip from the input_path
        clip = VideoFileClip(input_path)

        # Set the output path with .mp4 extension
        output_path_with_mp4 = os.path.splitext(output_path)[0] + ".mp4"

        # Convert and save the video clip to mp4 format
        clip.write_videofile(output_path_with_mp4, codec='libx264')

        print(f"Successfully converted {input_path} to {output_path_with_mp4}")
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")

def batch_convert_to_mp4(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)

        # Check if the file is a wav video
        if file_name.lower().endswith(".wmv"):
            # Set the output path with .mp4 extension
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp4")
            convert_to_mp4(input_path, output_path)

if __name__ == "__main__":
    input_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/video'
    output_folder = '/media/user/Disk1/chentingxiu/I2V/乳腺/rawframes'
    batch_convert_to_mp4(input_folder, output_folder)

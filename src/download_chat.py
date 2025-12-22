import subprocess
import os
import shutil

def download_and_rename(id_map_file, output_base_dir="data/raw"):
    """
    id_map_file: 一个文本文件，格式为 'video_id,custom_name'
    """
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    with open(id_map_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or ',' not in line:
                continue
            
            video_id, custom_name = [item.strip() for item in line.split(',')]
            
            # 1. 执行下载
            # tcd 会在 output_base_dir 下根据视频 ID 创建文件或文件夹
            print(f"--- Downloading ID: {video_id} (Target: {custom_name}) ---")
            try:
                subprocess.run([
                    "tcd",
                    "--video", video_id,
                    "--format", "json",
                    "--output", output_base_dir
                ], check=True)

                # 2. 自动定位下载的文件并重命名
                # tcd 通常生成的文件名包含 video_id，例如 '123456789.json'
                # 我们在目录下寻找匹配该 ID 的文件
                found = False
                for filename in os.listdir(output_base_dir):
                    if video_id in filename and filename.endswith(".json"):
                        old_path = os.path.join(output_base_dir, filename)
                        new_path = os.path.join(output_base_dir, f"{custom_name}.json")
                        
                        # 执行重命名（如果目标已存在则覆盖）
                        shutil.move(old_path, new_path)
                        print(f"Successfully renamed to: {new_path}")
                        found = True
                        break
                
                if not found:
                    print(f"Warning: Could not find downloaded file for ID {video_id}")

            except subprocess.CalledProcessError as e:
                print(f"Error downloading {video_id}: {e}")

if __name__ == "__main__":
    # 创建一个名为 video_list.txt 的文件，内容格式如下：
    # 789654123, stream_day_1
    # 987456321, stream_day_2
    download_and_rename("video_list.txt")
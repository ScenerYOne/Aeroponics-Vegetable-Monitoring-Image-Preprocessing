import os
import shutil
from pathlib import Path
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def merge_images_to_cam5(source_folder, destination_folder="cam5"):

    os.makedirs(destination_folder, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    total_copied = 0
    
    print(f"เริ่มรวมภาพจาก: {source_folder}")
    print(f"ไปยังโฟลเดอร์: {destination_folder}")
    print("-" * 50)
    
    # วนลูปผ่านทุกโฟลเดอร์และไฟล์
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                source_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, source_folder)
                if relative_path == ".":
                    new_filename = file
                else:
                    prefix = relative_path.replace(os.sep, "_")
                    name, ext = os.path.splitext(file)
                    new_filename = f"{prefix}_{name}{ext}"
                
                destination_path = os.path.join(destination_folder, new_filename)
                
                counter = 1
                while os.path.exists(destination_path):
                    name, ext = os.path.splitext(new_filename)
                    destination_path = os.path.join(destination_folder, f"{name}_{counter}{ext}")
                    counter += 1
                
                try:
                    shutil.copy2(source_path, destination_path)
                    total_copied += 1
                    print(f"✓ คัดลอก: {source_path} -> {destination_path}")
                except Exception as e:
                    print(f"✗ ข้อผิดพลาด: {source_path} - {str(e)}")
    
    print("-" * 50)
    print(f"เสร็จสิ้น คัดลอกภาพทั้งหมด {total_copied} ไฟล์")
    print(f"ภาพทั้งหมดอยู่ที่: {os.path.abspath(destination_folder)}")



if __name__ == "__main__":
    
    source = r"D:\cuu_hidro\result\cam5_panorama_24H"  
    merge_images_to_cam5(source)
    

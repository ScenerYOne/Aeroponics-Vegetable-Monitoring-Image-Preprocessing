import sys
import io

# Fix for UnicodeEncodeError on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import cv2
import os
import glob

# --- ตั้งค่า Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_INPUT_PATH = os.path.join(SCRIPT_DIR, 'result', 'cam5_bent_dual_24H')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'result', 'cam5_panorama_24H')

print(f"ค้นหาโฟลเดอร์ใน: {BASE_INPUT_PATH}")

# --- ค้นหาโฟลเดอร์ย่อย ---
try:
    all_items = os.listdir(BASE_INPUT_PATH)
    subfolders = [os.path.join(BASE_INPUT_PATH, item) for item in all_items 
                  if os.path.isdir(os.path.join(BASE_INPUT_PATH, item))]
    subfolders.sort()
except FileNotFoundError:
    print(f"ERROR: ไม่พบพาธ '{BASE_INPUT_PATH}'")
    sys.exit()

if not subfolders:
    print(f"ERROR: ไม่พบโฟลเดอร์ย่อย")
    sys.exit()

print(f"พบ {len(subfolders)} โฟลเดอร์\n")

# --- ฟังก์ชันสร้างการเบลนด์แบบ gradient ---
def blend_images_gradient(img_left, img_right, blend_width=50):
    """
    Parameters:
    - img_left: ภาพซ้าย (วางทางซ้าย)
    - img_right: ภาพขวา (วางทางขวา)
    - blend_width: ความกว้างของโซนเบลนด์ (pixels)

    """
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    
    if h_left != h_right:
        target_height = min(h_left, h_right)
        if h_left > target_height:
            img_left = cv2.resize(img_left, (w_left, target_height), interpolation=cv2.INTER_AREA)
        if h_right > target_height:
            img_right = cv2.resize(img_right, (w_right, target_height), interpolation=cv2.INTER_AREA)
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
    

    result_width = w_left + w_right - blend_width
    result = np.zeros((h_left, result_width, 3), dtype=np.uint8)
    
    result[:, :w_left] = img_left
    
    right_start = w_left - blend_width
    
    alpha = np.linspace(1, 0, blend_width).reshape(1, -1)
    alpha = np.repeat(alpha, h_left, axis=0)
    alpha = np.expand_dims(alpha, axis=2)
    alpha = np.repeat(alpha, 3, axis=2)

    blend_region_left = result[:, right_start:w_left].astype(float)
    blend_region_right = img_right[:, :blend_width].astype(float)
    
    blended = (blend_region_left * alpha + blend_region_right * (1 - alpha)).astype(np.uint8)
    result[:, right_start:w_left] = blended
    
    result[:, w_left:] = img_right[:, blend_width:]
    
    return result

# --- ฟังก์ชันต่อภาพแบบไม่มี blending (ต่อตรงๆ) ---
def concat_images_simple(img_left, img_right):

    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    
    if h_left != h_right:
        target_height = min(h_left, h_right)
        if h_left > target_height:
            img_left = cv2.resize(img_left, (w_left, target_height), interpolation=cv2.INTER_AREA)
        if h_right > target_height:
            img_right = cv2.resize(img_right, (w_right, target_height), interpolation=cv2.INTER_AREA)
    
    # ต่อภาพ
    result = cv2.hconcat([img_left, img_right])
    return result

# --- Loop ผ่านแต่ละโฟลเดอร์ ---
for folder_path in subfolders:
    folder_name = os.path.basename(os.path.normpath(folder_path))
    print(f"{'='*70}")
    print(f"กำลังประมวลผล: {folder_name}")
    print(f"{'='*70}")
    
    # Path ของโฟลเดอร์ left_bend และ right_bend
    left_folder = os.path.join(folder_path, 'left_bend')
    right_folder = os.path.join(folder_path, 'right_bend')
    
    if not os.path.exists(left_folder) or not os.path.exists(right_folder):
        print(f"ERROR: ไม่พบโฟลเดอร์ left_bend หรือ right_bend ใน {folder_name}")
        print(f"ข้ามโฟลเดอร์นี้\n")
        continue
    
    # ค้นหาไฟล์ภาพ
    left_files = sorted(glob.glob(os.path.join(left_folder, '*_left_bend.jpg')))
    right_files = sorted(glob.glob(os.path.join(right_folder, '*_right_bend.jpg')))
    
    if len(left_files) == 0 or len(right_files) == 0:
        print(f"ERROR: ไม่พบไฟล์ภาพใน {folder_name}")
        print(f"ข้ามโฟลเดอร์นี้...\n")
        continue
    
    print(f"พบภาพซ้าย: {len(left_files)} ไฟล์")
    print(f"พบภาพขวา: {len(right_files)} ไฟล์")
    
    # สร้างโฟลเดอร์ผลลัพธ์
    output_folder = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"บันทึกผลลัพธ์ที่: {output_folder}\n")
    
    # จับคู่และประมวลผล
    processed_count = 0
    
    for left_path in left_files:
        # ดึงชื่อไฟล์ฐาน (ไม่รวม _left_bend.jpg)
        base_name = os.path.basename(left_path).replace('_left_bend.jpg', '')
        
        # หาไฟล์ขวาที่ตรงกัน
        right_path = os.path.join(right_folder, f"{base_name}_right_bend.jpg")
        
        if not os.path.exists(right_path):
            print(f"  ⚠ ไม่พบคู่สำหรับ: {base_name}")
            continue
        
        # โหลดภาพ
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        
        if img_left is None or img_right is None:
            print(f"  ✗ ไม่สามารถโหลด: {base_name}")
            continue
        
        result = blend_images_gradient(img_right, img_left, blend_width=50)
        output_path = os.path.join(output_folder, f"{base_name}_panorama.jpg")
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        processed_count += 1
        
        if processed_count % 10 == 0 or processed_count == len(left_files):
            print(f"  ✓ ประมวลผล {processed_count}/{len(left_files)} ไฟล์")
    
    print(f"\nเสร็จสิ้นโฟลเดอร์ {folder_name}: ประมวลผล {processed_count} ไฟล์\n")

print(f"{'='*70}")
print("เสร็จสิ้นทุกโฟลเดอร์!")
print(f"ผลลัพธ์บันทึกที่: {OUTPUT_DIR}")
print(f"{'='*70}")
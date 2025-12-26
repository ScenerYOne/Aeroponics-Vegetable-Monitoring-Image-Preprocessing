# -*- coding: utf-8 -*-
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

# --- ค่าคงที่และตัวแปร Global ---
WINDOW_NAME = "Image - Click points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)"
PREVIEW_WINDOW = "Preview Transform Result"
points_src = []
g_transforms = []  # เก็บข้อมูล transform ทั้งหมด

# --- ฟังก์ชันคำนวณขนาดผลลัพธ์จากจุด 4 จุด ---
def calculate_output_size(pts):
    pt_TL = pts[0]
    pt_TR = pts[1]
    pt_BR = pts[2]
    pt_BL = pts[3]
    
    width_top = np.linalg.norm(pt_TR - pt_TL)
    width_bottom = np.linalg.norm(pt_BR - pt_BL)
    maxWidth = max(int(width_top), int(width_bottom))
    
    height_left = np.linalg.norm(pt_BL - pt_TL)
    height_right = np.linalg.norm(pt_BR - pt_TR)
    maxHeight = max(int(height_left), int(height_right))
    
    print(f"    Width: top={width_top:.1f}, bottom={width_bottom:.1f} -> {maxWidth}")
    print(f"    Height: left={height_left:.1f}, right={height_right:.1f} -> {maxHeight}")
    
    return maxWidth, maxHeight

# --- ฟังก์ชันสร้างภาพแบบ Enhanced Focus (เวอร์ชันปรับปรุง - ไม่มีขอบ) ---
def create_enhanced_focus_image(img_original, pts_src, matrix, output_size, margin_ratio=0.35):
    """
    สร้างภาพที่ส่วนกลาง (Transform) ชัดเจน และส่วนข้างบีบแบบสมูท
    ใช้เทคนิค multi-band blending เพื่อไม่ให้เห็นขอบ
    """
    h_orig, w_orig = img_original.shape[:2]
    
    # หาขอบเขตของจุดที่เลือก
    x_coords = [pt[0] for pt in pts_src]
    y_coords = [pt[1] for pt in pts_src]
    
    x_min = int(min(x_coords))
    x_max = int(max(x_coords))
    y_min = int(min(y_coords))
    y_max = int(max(y_coords))
    
    # ทำ Perspective Transform
    transformed = cv2.warpPerspective(img_original, matrix, output_size, 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
    transform_h = output_size[1]
    
    left_margin_width = int(output_size[0] * margin_ratio)
    right_margin_width = int(output_size[0] * margin_ratio)
    
    left_part = None
    if x_min > 10:
        y_start = max(0, int(y_min))
        y_end = min(h_orig, int(y_max))
        
        left_region = img_original[y_start:y_end, max(0, x_min-50):x_min]
        
        if left_region.shape[0] > 0 and left_region.shape[1] > 0:
            left_part = cv2.resize(left_region, 
                                  (left_margin_width, transform_h), 
                                  interpolation=cv2.INTER_LINEAR)
            left_part = cv2.bilateralFilter(left_part, 5, 50, 50)
    
    right_part = None
    if x_max < w_orig - 10:
        y_start = max(0, int(y_min))
        y_end = min(h_orig, int(y_max))
        
        right_region = img_original[y_start:y_end, x_max:min(w_orig, x_max+50)]
        
        if right_region.shape[0] > 0 and right_region.shape[1] > 0:
            right_part = cv2.resize(right_region, 
                                   (right_margin_width, transform_h), 
                                   interpolation=cv2.INTER_LINEAR)
            right_part = cv2.bilateralFilter(right_part, 5, 50, 50)
    
    # === รวมภาพด้วย Gradient Blending ===
    parts = []
    if left_part is not None and left_part.size > 0:
        parts.append(left_part)
    parts.append(transformed)
    if right_part is not None and right_part.size > 0:
        parts.append(right_part)
    
    if len(parts) == 1:
        return parts[0]
    
    # รวมภาพพื้นฐาน
    result = cv2.hconcat(parts)
    
    feather_width = 25
    
    if len(parts) >= 2:
        seam_x_list = []
        current_x = 0
        for i in range(len(parts) - 1):
            current_x += parts[i].shape[1]
            seam_x_list.append(current_x)

        for seam_x in seam_x_list:
            start_x = max(0, seam_x - feather_width)
            end_x = min(result.shape[1], seam_x + feather_width)
            
            if end_x - start_x < 2:
                continue
            
            for i in range(end_x - start_x):
                x_pos = start_x + i
                if x_pos <= 0 or x_pos >= result.shape[1] - 1:
                    continue
                
                relative_pos = (x_pos - start_x) / (end_x - start_x)
                if relative_pos < 0.5:
                    alpha = relative_pos * 2  # 0 -> 1
                else:
                    alpha = (1 - relative_pos) * 2  # 1 -> 0
                
                # Blur pixel นี้
                if alpha < 0.99:
                    col = result[:, x_pos:x_pos+1].copy()
                    col_blurred = cv2.GaussianBlur(col, (1, 11), 0)
                    result[:, x_pos:x_pos+1] = cv2.addWeighted(
                        col, alpha, col_blurred, 1-alpha, 0
                    )
    
    return result

# --- ฟังก์ชันย่อ/ขยายภาพโดยรักษาสัดส่วน ---
def resize_image(image, max_dim):
    h, w = image.shape[:2]
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# --- Mouse Callback ---
def mouse_callback(event, x, y, flags, param):
    global points_src

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_src) >= 12:
            print("คลิกครบ 12 จุด กด 'p'ดูตัวอย่าง หรือ 'y'ยืนยัน")
            return

        resize_ratio = param['resize_ratio']
        x_orig = int(x / resize_ratio)
        y_orig = int(y / resize_ratio)
        
        points_src.append((x_orig, y_orig))

        img_display = param['image']
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)

        point_num = len(points_src)
        color = (0, 0, 255) if point_num % 4 != 0 else (255, 0, 0)
        cv2.putText(img_display, str(point_num), (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        group_start = ((point_num - 1) // 4) * 4
        if point_num > group_start + 1:
            prev_idx = point_num - 2
            prev_x = int(points_src[prev_idx][0] * resize_ratio)
            prev_y = int(points_src[prev_idx][1] * resize_ratio)
            cv2.line(img_display, (prev_x, prev_y), (x, y), (0, 255, 255), 2)

        if point_num % 4 == 0:
            first_idx = group_start
            first_x = int(points_src[first_idx][0] * resize_ratio)
            first_y = int(points_src[first_idx][1] * resize_ratio)
            cv2.line(img_display, (x, y), (first_x, first_y), (0, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, img_display)
        
        position_name = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"][(point_num - 1) % 4]
        print(f"  จุดที่ {point_num} ({position_name}): x={x_orig}, y={y_orig}")

# --- ฟังก์ชันแสดงตัวอย่างผลลัพธ์ ---
def show_preview(img_original, resize_ratio_display):
    """แสดงตัวอย่างผลลัพธ์"""
    if len(points_src) < 4:
        print("ต้องมีอย่างน้อย 4 จุดเพื่อแสดงตัวอย่าง")
        return
    
    num_sets = len(points_src) // 4
    
    for i in range(num_sets):
        start_idx = i * 4
        end_idx = start_idx + 4
        pts_src = np.float32(points_src[start_idx:end_idx])
        
        width, height = calculate_output_size(pts_src)
        pts_dst = np.float32([
            [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
        ])
        
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        
        composite = create_enhanced_focus_image(img_original, pts_src, matrix, (width, height))

        preview = resize_image(composite, 900)

        section_name = ["Left", "Middle", "Right"][i]
        cv2.putText(preview, f"{section_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"{PREVIEW_WINDOW} - {section_name}", preview)
    
    print("\nแสดงตัวอย่างผลลัพธ์  'y' ยืนยัน หรือ 'c' แก้ไข")

# --- ฟังก์ชันคำนวณ Matrix ---
def process_and_calculate_matrices():
    global points_src, g_transforms

    num_points = len(points_src)
    if num_points < 4:
        print("ERROR: ต้องมีอย่างน้อย 4 จุด!")
        return False

    print(f"\nกำลังคำนวณ Perspective Matrix จาก {num_points} จุด...")
    
    g_transforms = []
    num_sets = num_points // 4

    for i in range(num_sets):
        start_idx = i * 4
        end_idx = start_idx + 4
        pts_src = np.float32(points_src[start_idx:end_idx])
        
        width, height = calculate_output_size(pts_src)
        pts_dst = np.float32([
            [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
        ])
        
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        
        transform_data = {
            'side': ['left', 'middle', 'right'][i],
            'points': pts_src,
            'matrix': matrix,
            'output_size': (width, height)
        }
        g_transforms.append(transform_data)
        
        print(f"  ✓ Matrix {i+1} ({transform_data['side']}): {width}x{height}")
    
    return True

# --- 1. กำหนด Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, 'data', 'cam5')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'result', 'cam5_transformed')

print(f"ค้นหาโฟลเดอร์ใน: {BASE_PATH}")

# --- 2. ค้นหาโฟลเดอร์ย่อย ---
try:
    all_items = os.listdir(BASE_PATH)
    subfolders = [os.path.join(BASE_PATH, item) for item in all_items 
                  if os.path.isdir(os.path.join(BASE_PATH, item))]
    subfolders.sort()
except FileNotFoundError:
    print(f"ERROR: ไม่พบพาธ '{BASE_PATH}'")
    sys.exit()

if not subfolders:
    print(f"ERROR: ไม่พบโฟลเดอร์ย่อย")
    sys.exit()

print(f"พบ {len(subfolders)} โฟลเดอร์")

# --- 3. Loop หลัก ---
for folder_path in subfolders:
    folder_name = os.path.basename(os.path.normpath(folder_path))
    print(f"\n{'='*70}")
    print(f"โฟลเดอร์: {folder_name}")
    print(f"{'='*70}")
    
    # ค้นหาไฟล์
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        print(f"ไม่พบไฟล์ .jpg ข้าม")
        continue
    
    print(f"พบ {len(image_files)} ไฟล์")

    # รีเซ็ตตัวแปร
    points_src = []
    g_transforms = []
    
    # โหลดภาพตัวอย่าง
    SAMPLE_IMAGE = image_files[0]
    img_setup = cv2.imread(SAMPLE_IMAGE)
    if img_setup is None:
        print(f"ERROR: ไม่สามารถโหลดภาพ")
        continue
    
    # ย่อขนาดสำหรับแสดงผล
    max_display = 1000
    img_display = resize_image(img_setup.copy(), max_display)
    h_orig, w_orig = img_setup.shape[:2]
    h_disp, w_disp = img_display.shape[:2]
    resize_ratio = w_disp / w_orig

    # Setup mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, 
                         {'image': img_display, 'resize_ratio': resize_ratio})

    print(f"\nภาพตัวอย่าง: {os.path.basename(SAMPLE_IMAGE)}")
    print(f"ขนาดต้นฉบับ: {w_orig}x{h_orig}")
    print("\n" + "="*70)
    print("  - คลิกตามลำดับ: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
    print("  - คลิกได้ 4, 8, หรือ 12 จุด")
    print("\nปุ่มควบคุม:")
    print("  [p] = แสดงตัวอย่างผลลัพธ์")
    print("  [y] = ยืนยันและเริ่มประมวลผล")
    print("  [c] = ล้างและเริ่มใหม่")
    print("  [q] = ข้ามโฟลเดอร์นี้")
    print("="*70 + "\n")
    
    cv2.imshow(WINDOW_NAME, img_display)
    
    # Loop รอการกดปุ่ม
    while True:
        key = cv2.waitKey(1) & 0xFF

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("ปิดหน้าต่าง ข้าม...")
            g_transforms = []
            break

        if key == ord('p'):
            show_preview(img_setup, resize_ratio)
        
        elif key == ord('y'):
            if process_and_calculate_matrices():
                print("ยืนยัน กำลังประมวลผล")
                cv2.waitKey(1000)
            break
        
        elif key == ord('c'):
            print("\nล้างทั้งหมด เริ่มใหม่")
            points_src = []
            img_display = resize_image(img_setup.copy(), max_display)
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, 
                               {'image': img_display, 'resize_ratio': resize_ratio})
            cv2.imshow(WINDOW_NAME, img_display)
            try:
                cv2.destroyWindow(f"{PREVIEW_WINDOW} - Left")
                cv2.destroyWindow(f"{PREVIEW_WINDOW} - Middle")
                cv2.destroyWindow(f"{PREVIEW_WINDOW} - Right")
            except:
                pass

        elif key == ord('q'):
            print("ข้ามโฟลเดอร์นี้")
            g_transforms = []
            break

    cv2.destroyAllWindows()

    if not g_transforms:
        print(f"ข้ามโฟลเดอร์ {folder_name}")
        continue

    # สร้างโฟลเดอร์ผลลัพธ์
    output_base = os.path.join(OUTPUT_DIR, folder_name)
    output_folders = {}
    
    for transform_data in g_transforms:
        side = transform_data['side']
        folder_path = os.path.join(output_base, side)
        os.makedirs(folder_path, exist_ok=True)
        output_folders[side] = folder_path

    print(f"\nโฟลเดอร์ผลลัพธ์:")
    for side, path in output_folders.items():
        print(f"  - {side}: {path}")

    # ประมวลผลภาพทั้งหมด
    print(f"\nเริ่มแปลงภาพ {len(image_files)} ไฟล์\n")
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ✗ ไม่สามารถอ่าน: {os.path.basename(img_path)}")
            continue
        
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        
        for transform_data in g_transforms:
            side = transform_data['side']
            pts = transform_data['points']
            matrix = transform_data['matrix']
            output_size = transform_data['output_size']
            composite = create_enhanced_focus_image(img, pts, matrix, output_size)
            
            save_path = os.path.join(output_folders[side], f"{base_filename}_{side}.jpg")
            cv2.imwrite(save_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"  ✓ ประมวลผล {i+1}/{len(image_files)} ไฟล์")

    print(f"\n{'='*70}")
    print(f"เสร็จสิ้นโฟลเดอร์ {folder_name}")
    print(f"{'='*70}")

print(f"\n{'='*70}")
print("เสร็จสิ้นทุกโฟลเดอร์!")
print(f"{'='*70}")
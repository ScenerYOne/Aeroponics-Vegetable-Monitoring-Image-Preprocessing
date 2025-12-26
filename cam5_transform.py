import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import numpy as np
import cv2
import os
import glob


WINDOW_NAME = "Image - Click 4 points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)"
PREVIEW_WINDOW = "Preview Transform Result"
points_src = []
g_transforms = [] 


def calculate_output_size(pts):
    pt_TL = pts[0]
    pt_TR = pts[1]
    pt_BR = pts[2]
    pt_BL = pts[3]
    
    width_top = np.linalg.norm(pt_TR - pt_TL)
    width_bottom = np.linalg.norm(pt_BR - pt_BL)
    maxWidth = max(1, int(width_top), int(width_bottom))
    
    height_left = np.linalg.norm(pt_BL - pt_TL)
    height_right = np.linalg.norm(pt_BR - pt_TR)
    maxHeight = max(1, int(height_left), int(height_right))
    
    return maxWidth, maxHeight

def create_cropped_transform(img_original, matrix, output_size):
    """
    ทำ Perspective Transform โดยไม่เพิ่มขอบดำรอบภาพ
    """
    transformed = cv2.warpPerspective(img_original, matrix, output_size, 
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
    return transformed


def resize_image(image, max_dim):
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def mouse_callback(event, x, y, flags, param):
    global points_src

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_src) >= 4:
            print("คลิกครบ 4 จุดแล้ว กด 'p' ดูตัวอย่าง  'y' ยืนยัน")
            return

        resize_ratio = param['resize_ratio']
        x_orig = int(x / resize_ratio)
        y_orig = int(y / resize_ratio)
        
        points_src.append((x_orig, y_orig))
        
        img_display = param['image']
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        
        point_num = len(points_src)
        position_name = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"][(point_num - 1) % 4]
        color = (0, 0, 255)
        cv2.putText(img_display, str(point_num), (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if point_num > 1:
            prev_idx = point_num - 2
            prev_x = int(points_src[prev_idx][0] * resize_ratio)
            prev_y = int(points_src[prev_idx][1] * resize_ratio)
            cv2.line(img_display, (prev_x, prev_y), (x, y), (0, 255, 255), 2)
        
        if point_num == 4:
            first_idx = 0
            first_x = int(points_src[first_idx][0] * resize_ratio)
            first_y = int(points_src[first_idx][1] * resize_ratio)
            cv2.line(img_display, (x, y), (first_x, first_y), (0, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, img_display)
        
        print(f"   จุดที่ {point_num} ({position_name}): x={x_orig}, y={y_orig}")


def create_bent_destination_points(width, height, bend_direction='left', bend_factor=0.25):
    """
    ปรับให้ output อยู่ใน positive space 
    """
    h_shift = int(width * bend_factor)
    v_shift = int(height * 0.08)
    
    if bend_direction == 'right':
        
        pts_dst = np.float32([
            [0, v_shift],                          
            [width - h_shift, 0],                  
            [width - h_shift, height],             
            [0, height - v_shift]                  
        ])
    else:  
       
        pts_dst = np.float32([
            [h_shift, 0],                          
            [width, v_shift],                      
            [width, height - v_shift],             
            [h_shift, height]                      
        ])
    
    return pts_dst


def show_preview(img_original, resize_ratio_display):
    if len(points_src) != 4:
        print("ต้องมี 4 จุดพอดีเพื่อแสดงตัวอย่าง")
        return
    
    pts_src = np.float32(points_src)
    base_width, base_height = calculate_output_size(pts_src)
    
    
    pts_dst_left = create_bent_destination_points(base_width, base_height, 'left', bend_factor=0.25)
    
    
    min_x = np.min(pts_dst_left[:, 0])
    min_y = np.min(pts_dst_left[:, 1])
    max_x = np.max(pts_dst_left[:, 0])
    max_y = np.max(pts_dst_left[:, 1])
    
   
    output_width_left = int(np.ceil(max_x - min_x))
    output_height_left = int(np.ceil(max_y - min_y))
    output_size_left = (output_width_left, output_height_left)
    
    
    pts_dst_left_adjusted = pts_dst_left - [min_x, min_y]
    
    matrix_left = cv2.getPerspectiveTransform(pts_src, pts_dst_left_adjusted)
    composite_left = create_cropped_transform(img_original, matrix_left, output_size_left)
    
    preview_left = resize_image(composite_left, 900)
    cv2.putText(preview_left, "Preview - Left Bend (Cropped)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(f"{PREVIEW_WINDOW} - Left", preview_left)

    
    pts_dst_right = create_bent_destination_points(base_width, base_height, 'right', bend_factor=0.25)
    
    min_x = np.min(pts_dst_right[:, 0])
    min_y = np.min(pts_dst_right[:, 1])
    max_x = np.max(pts_dst_right[:, 0])
    max_y = np.max(pts_dst_right[:, 1])
    
    output_width_right = int(np.ceil(max_x - min_x))
    output_height_right = int(np.ceil(max_y - min_y))
    output_size_right = (output_width_right, output_height_right)
    
    pts_dst_right_adjusted = pts_dst_right - [min_x, min_y]
    
    matrix_right = cv2.getPerspectiveTransform(pts_src, pts_dst_right_adjusted)
    composite_right = create_cropped_transform(img_original, matrix_right, output_size_right)
    
    preview_right = resize_image(composite_right, 900)
    cv2.putText(preview_right, "Preview - Right Bend (Cropped)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(f"{PREVIEW_WINDOW} - Right", preview_right)
    
    print("\nแสดงตัวอย่างผลลัพธ์ 'y' ยืนยัน หรือ 'c' แก้ไข")

# --- ฟังก์ชันคำนวณ Matrix  ---
def process_and_calculate_matrices():
    global points_src, g_transforms

    num_points = len(points_src)
    if num_points != 4:
        print(f"ERROR: ต้องคลิก 4 จุดพอดี! (คลิกไป {num_points} จุด)")
        return False

    print(f"\nกำลังคำนวณ Perspective Matrix จาก 4 จุด...")
    
    g_transforms = []
    pts_src = np.float32(points_src)
    base_width, base_height = calculate_output_size(pts_src)
    
    # --- สำหรับภาพบิดซ้าย ---
    pts_dst_left = create_bent_destination_points(base_width, base_height, 'left', bend_factor=0.25)
    
    min_x = np.min(pts_dst_left[:, 0])
    min_y = np.min(pts_dst_left[:, 1])
    max_x = np.max(pts_dst_left[:, 0])
    max_y = np.max(pts_dst_left[:, 1])
    
    output_width_left = int(np.ceil(max_x - min_x))
    output_height_left = int(np.ceil(max_y - min_y))
    output_size_left = (output_width_left, output_height_left)
    
    pts_dst_left_adjusted = pts_dst_left - [min_x, min_y]
    matrix_left = cv2.getPerspectiveTransform(pts_src, pts_dst_left_adjusted)
    
    g_transforms.append({
        'side': 'left_bend',
        'points': pts_src,
        'matrix': matrix_left,
        'output_size': output_size_left
    })
    print(f"   ✓ Matrix (left_bend): {output_size_left[0]}x{output_size_left[1]}")

    # --- สำหรับภาพบิดขวา ---
    pts_dst_right = create_bent_destination_points(base_width, base_height, 'right', bend_factor=0.25)
    
    min_x = np.min(pts_dst_right[:, 0])
    min_y = np.min(pts_dst_right[:, 1])
    max_x = np.max(pts_dst_right[:, 0])
    max_y = np.max(pts_dst_right[:, 1])
    
    output_width_right = int(np.ceil(max_x - min_x))
    output_height_right = int(np.ceil(max_y - min_y))
    output_size_right = (output_width_right, output_height_right)
    
    pts_dst_right_adjusted = pts_dst_right - [min_x, min_y]
    matrix_right = cv2.getPerspectiveTransform(pts_src, pts_dst_right_adjusted)
    
    g_transforms.append({
        'side': 'right_bend',
        'points': pts_src,
        'matrix': matrix_right,
        'output_size': output_size_right
    })
    print(f"   ✓ Matrix (right_bend): {output_size_right[0]}x{output_size_right[1]}")
    
    return True

# --- 1. กำหนด Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, 'data', 'cam5_24H',)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'result', 'cam5_bent_dual_24H')

print(f"ค้นหาโฟลเดอร์ใน: {BASE_PATH}")

# --- 2. ค้นหาโฟลเดอร์ย่อย ---
try:
    all_items = os.listdir(BASE_PATH)
    subfolders = [os.path.join(BASE_PATH, item) for item in all_items 
                  if os.path.isdir(os.path.join(BASE_PATH, item))]
    subfolders.sort()
except FileNotFoundError:
    print(f"ERROR: ไม่พบpath '{BASE_PATH}'")
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
    
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        print(f"ไม่พบไฟล์ .jpg ข้าม")
        continue
    
    print(f"พบ {len(image_files)} ไฟล์")

    points_src = []
    g_transforms = []
    
    SAMPLE_IMAGE = image_files[0]
    img_setup = cv2.imread(SAMPLE_IMAGE)
    if img_setup is None:
        print(f"ERROR: ไม่สามารถโหลดภาพ '{SAMPLE_IMAGE}' ได้")
        continue
    
    max_display = 1000
    img_display = resize_image(img_setup.copy(), max_display)
    h_orig, w_orig = img_setup.shape[:2]
    h_disp, w_disp = img_display.shape[:2]
    resize_ratio = 1.0 if w_orig == 0 else w_disp / w_orig

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, 
                         {'image': img_display, 'resize_ratio': resize_ratio})

    print(f"\nภาพตัวอย่าง: {os.path.basename(SAMPLE_IMAGE)}")
    print(f"ขนาดต้นฉบับ: {w_orig}x{h_orig}")
    print("\n" + "="*70)
    print("   - คลิก 4 จุดบนพื้นที่ที่ต้องการแปลง ตามลำดับ:")
    print("     Top-Left → Top-Right → Bottom-Right → Bottom-Left")
    print("\nปุ่มควบคุม:")
    print("   [p] = แสดงตัวอย่างผลลัพธ์")
    print("   [y] = ยืนยันและเริ่มประมวลผล")
    print("   [c] = ล้างและเริ่มใหม่")
    print("   [q] = ข้ามโฟลเดอร์นี้")
    print("="*70 + "\n")
    
    cv2.imshow(WINDOW_NAME, img_display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("ปิดหน้าต่าง ข้าม")
            g_transforms = []
            break

        if key == ord('p'):
            show_preview(img_setup, resize_ratio)
        
        elif key == ord('y'):
            if process_and_calculate_matrices():
                print("ยืนยัน ")
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
                cv2.destroyWindow(f"{PREVIEW_WINDOW} - Right")
            except:
                pass

        elif key == ord('q'):
            print("ข้ามโฟลเดอร์")
            g_transforms = []
            break

    cv2.destroyAllWindows()

    if not g_transforms:
        print(f"ข้ามโฟลเดอร์ {folder_name}")
        continue

    output_base = os.path.join(OUTPUT_DIR, folder_name)
    output_folders = {}
    
    for transform_data in g_transforms:
        side = transform_data['side']
        folder_path = os.path.join(output_base, side)
        os.makedirs(folder_path, exist_ok=True)
        output_folders[side] = folder_path

    print(f"\nโฟลเดอร์ผลลัพธ์:")
    for side, path in output_folders.items():
        print(f"   - {side}: {path}")

    print(f"\nเริ่มแปลงภาพ {len(image_files)} ไฟล์\n")
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"   ✗ ไม่สามารถอ่าน: {os.path.basename(img_path)}")
            continue
        
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        for transform_data in g_transforms:
            side = transform_data['side']
            matrix = transform_data['matrix']
            output_size = transform_data['output_size']
            
            composite = create_cropped_transform(img, matrix, output_size)
            
            save_path = os.path.join(output_folders[side], f"{base_filename}_{side}.jpg")
            cv2.imwrite(save_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"   ✓ ประมวลผล {i+1}/{len(image_files)} ไฟล์")

    print(f"\n{'='*70}")
    print(f"เสร็จสิ้นโฟลเดอร์ {folder_name}")
    print(f"{'='*70}")

print(f"\n{'='*70}")
print("เสร็จสิ้นทุกโฟลเดอร์")
print(f"{'='*70}")
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import sys
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
from collections import deque


# ---------------- 自定义双缓冲标签（解决UI闪烁核心） ----------------
class DoubleBufferLabel(tk.Label):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # The black background reduces the flickering effect
        self.config(bg='black')
        self.imgtk = None

    def update_image(self, img):
        self.imgtk = img
        self.configure(image=self.imgtk)
        self.update_idletasks()


# ---------------- 初始化 ----------------
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# 剪刀手检测变量
last_v_sign_time = 0
v_sign_duration = 0.0
V_SIGN_HOLD = 1.0  # 剪刀手持续秒数


# 笑脸检测变量
last_smile_time = 0
smile_duration = 0.0
SMILE_HOLD = 1.0  # 笑脸持续秒数


# MediaPipe实例统一初始化（复用，避免重复创建）
face_mesh = mp_face.FaceMesh(max_num_faces=5)
hands = mp_hands.Hands()
face_detect = mp_face_detection.FaceDetection()  # 关键：只创建1次

# 文件夹创建
if not os.path.exists("photos"):
    os.mkdir("photos")

# 全局变量
cap = None
running = False
last_capture_time = 0
current_frame = None
frame_buffer = deque(maxlen=2)  # 帧缓冲队列，解决帧率波动
beauty_enabled = True
eye_enabled = False   # 👈 默认关闭大眼
v_sign_frames = 0
smile_start_time = 0
last_photo_path = None
recent_photos = deque(maxlen=4)   # 👈 最近两张


# ---------------- 功能函数 ----------------
def slim_face(frame, face_landmarks, strength=0.04):
    h, w = frame.shape[:2]

    # 下颌轮廓点（左右）
    jaw_ids = list(range(234, 244)) + list(range(454, 464))

    for idx in jaw_ids:
        p = face_landmarks.landmark[idx]
        cx, cy = int(p.x * w), int(p.y * h)

        # 向人脸中心移动
        nx = int(cx + (w//2 - cx) * strength)
        ny = cy

        cv2.circle(frame, (nx, ny), 1, (0,0,0), -1)

    return frame

def enlarge_eyes(frame, face_landmarks, strength=0.18):
    h, w = frame.shape[:2]
    result = frame.copy()

    left_eye_ids  = [33, 133, 159, 145]
    right_eye_ids = [362, 263, 386, 374]

    for eye_ids in [left_eye_ids, right_eye_ids]:
        cx = int(np.mean([face_landmarks.landmark[i].x for i in eye_ids]) * w)
        cy = int(np.mean([face_landmarks.landmark[i].y for i in eye_ids]) * h)

        r = int(0.045 * w)   # 👈 半径缩小

        for y in range(cy - r, cy + r):
            for x in range(cx - r, cx + r):
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue

                dx = x - cx
                dy = y - cy
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < r:
                    scale = 1 - strength * (1 - dist / r)
                    src_x = int(cx + dx * scale)
                    src_y = int(cy + dy * scale)

                    result[y, x] = frame[src_y, src_x]

    return result


def toggle_eye():
    global eye_enabled
    eye_enabled = not eye_enabled
    btn_eye.config(
        text="关闭大眼" if eye_enabled else "开启大眼"
    )

def get_skin_mask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)

    # 经典肤色阈值（亚洲人群非常稳）
    skin_mask = cv2.inRange(
        ycrcb,
        (0, 133, 77),
        (255, 173, 127)
    )

    # 去噪 + 平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)

    return skin_mask

def get_face_region_mask(frame, detections):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not detections:
        return mask

    for det in detections:
        bbox = det.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        # 向下扩一点（包含下巴，不包含衣服）
        y2 = min(h, int(y2 + 0.15 * (y2 - y1)))

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask

def get_face_mesh_mask(frame, face_landmarks):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # MediaPipe 脸部轮廓点
    face_oval_ids = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]

    points = []
    for idx in face_oval_ids:
        p = face_landmarks.landmark[idx]
        points.append((int(p.x * w), int(p.y * h)))

    cv2.fillPoly(mask, [np.array(points)], 255)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    return mask


def beautify(frame, face_result, face_detections, enable=True):
    if not enable:
        return frame

    # 1. 轻磨皮（保留纹理）
    smooth = cv2.bilateralFilter(frame, 9, 60, 60)
    frame = cv2.addWeighted(frame, 0.4, smooth, 0.6, 0)

    # ===============================
    # 2. 皮肤区域美白（只对白脸）
    # ===============================

    # 2.1 基础肤色 mask（颜色）
    skin_mask = get_skin_mask(frame)

    # 2.2 人脸区域 mask（排除衣服）
    face_region_mask = get_face_region_mask(frame, face_detections)

    # 2.3 融合：肤色 AND 人脸区域
    skin_mask = cv2.bitwise_and(skin_mask, face_region_mask)

    # 2.4 进一步用 FaceMesh 精确限制（脸轮廓）
    if face_result.multi_face_landmarks:
        mesh_mask_total = np.zeros_like(skin_mask)
        for face_landmarks in face_result.multi_face_landmarks:
            mesh_mask = get_face_mesh_mask(frame, face_landmarks)
            mesh_mask_total = cv2.bitwise_or(mesh_mask_total, mesh_mask)

        skin_mask = cv2.bitwise_and(skin_mask, mesh_mask_total)

    # 2.5 LAB 空间定向美白
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_white = cv2.add(l, 15)          # 美白强度（15~18 推荐）
    l_white = np.clip(l_white, 0, 255)

    l = np.where(skin_mask > 0, l_white, l)

    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ===============================
    # 3. 瘦脸 / 大眼
    # ===============================
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            frame = slim_face(frame, face_landmarks)
            if eye_enabled:
                frame = enlarge_eyes(frame, face_landmarks)

    # ===============================
    # 4. 去噪 + 防糊
    # ===============================
    frame = cv2.medianBlur(frame, 3)

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    frame = cv2.filter2D(frame, -1, sharpen_kernel)

    return frame


def toggle_beauty():
    global beauty_enabled
    beauty_enabled = not beauty_enabled
    btn_beauty.config(
        text="关闭美颜" if beauty_enabled else "开启美颜"
    )


def is_smiling(face_result):
    if not face_result.multi_face_landmarks:
        return False

    for face_landmarks in face_result.multi_face_landmarks:
        top = face_landmarks.landmark[13].y
        bottom = face_landmarks.landmark[14].y
        if (bottom - top) > 0.012:  # 👈 threshold value

            return True

    return False

# ---------- 1. 替换剪刀手判断 ----------
def is_v_sign(hand_result):
    """
    稳定剪刀手：仅食指+中指伸直，其余弯曲；两指夹角≈15°-65°；掌心朝相机
    """
    if not hand_result.multi_hand_landmarks:
        return False
    for hand_landmarks in hand_result.multi_hand_landmarks:
        lm = hand_landmarks.landmark
        h, w = 480, 640          # 与 capture 设置一致，用于像素坐标

        # 0-20 关键点
        def get_xy(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        # 指尖
        t_idx = get_xy(8)   # 食指
        t_mid = get_xy(12)  # 中指
        t_rin = get_xy(16)  # 无名指
        t_pnk = get_xy(20)  # 小指
        # 指根
        b_idx = get_xy(5)
        b_mid = get_xy(9)
        b_rin = get_xy(13)
        b_pnk = get_xy(17)

        # 1. Straightening judgment: The y at the fingertip is less than
        # the y at the base of the finger (higher on the image)
        straight_idx = t_idx[1] < b_idx[1]
        straight_mid = t_mid[1] < b_mid[1]
        straight_rin = t_rin[1] < b_rin[1]
        straight_pnk = t_pnk[1] < b_pnk[1]

        # 2. Only keep the index finger and middle finger straight
        if not (straight_idx and straight_mid):
            continue
        if straight_rin or straight_pnk:
            continue

        # 3. The Angle between two fingers ≈ 15-65° (dot product of vectors)
        vec_idx = np.array(t_idx) - np.array(b_idx)
        vec_mid = np.array(t_mid) - np.array(b_mid)
        cos_angle = np.dot(vec_idx, vec_mid) / (np.linalg.norm(vec_idx) * np.linalg.norm(vec_mid) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        if 15 < angle < 65:
            return True
    return False

def show_auto_message(text, duration=2000):
    win = tk.Toplevel(root)
    win.overrideredirect(True)   # 无边框
    win.attributes("-topmost", True)

    label = tk.Label(
        win, text=text,
        bg="#333", fg="white",
        font=("Arial", 12),
        padx=20, pady=10
    )
    label.pack()

    # 居中显示
    win.update_idletasks()
    x = root.winfo_x() + root.winfo_width() // 2 - win.winfo_width() // 2
    y = root.winfo_y() + root.winfo_height() // 2 - win.winfo_height() // 2
    win.geometry(f"+{x}+{y}")

    win.after(duration, win.destroy)

def update_preview_images():
    for i, lbl in enumerate(preview_labels):
        if i < len(recent_photos):
            img = Image.open(recent_photos[i])
            img = img.resize((180, 120))
            imgtk = ImageTk.PhotoImage(img)

            lbl.imgtk = imgtk   # 防止 GC
            lbl.config(image=imgtk)
        else:
            lbl.config(image='', bg="#111")



def load_last_photo_from_disk():
    global last_photo_path

    if not os.path.exists("photos"):
        return

    files = [
        os.path.join("photos", f)
        for f in os.listdir("photos")
        if f.lower().endswith(".jpg")
    ]

    if not files:
        return

    # 按修改时间排序，取最新
    last_photo_path = max(files, key=os.path.getmtime)
    update_preview_images()



def play_shutter_sound():
    try:
        root.bell()  # Tk 自带提示音，类似“咔”
    except:
        pass


def capture_photo(auto=False):
    global current_frame, last_photo_path

    if current_frame is None:
        return

    filename = f"photos/{'auto' if auto else 'manual'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, current_frame)

    play_shutter_sound()
    show_auto_message("📸 拍照成功", 2000)

    # last_photo_path = filename

    # ✅ 最新照片放最前面
    recent_photos.appendleft(filename)

    update_preview_images()

def show_large_image(img_path):
    if not os.path.exists(img_path):
        return

    win = tk.Toplevel(root)
    win.title("照片预览")
    win.configure(bg="black")

    img = Image.open(img_path)

    # 限制最大尺寸，防止超屏
    max_w, max_h = 900, 700
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1)
    img = img.resize((int(w * scale), int(h * scale)))

    imgtk = ImageTk.PhotoImage(img)

    lbl = tk.Label(win, image=imgtk, bg="black")
    lbl.imgtk = imgtk
    lbl.pack(padx=10, pady=10)



def open_album():
    """打开照片文件夹（保持原逻辑）"""
    path = os.path.abspath("photos")
    if os.name == "nt":
        os.startfile(path)
    else:
        subprocess.Popen(["open", path])

def load_last_photos_from_disk():
    if not os.path.exists("photos"):
        return

    files = [
        os.path.join("photos", f)
        for f in os.listdir("photos")
        if f.lower().endswith(".jpg")
    ]

    if not files:
        return

    # 按修改时间排序（最新在前）
    files.sort(key=os.path.getmtime, reverse=True)

    recent_photos.clear()
    for f in files[:4]:
        recent_photos.append(f)

    update_preview_images()



def open_camera():
    """打开相机（新增：快门同步+固定分辨率+自动曝光优化）"""
    global cap, running
    if running:
        messagebox.showwarning("提示", "相机已经打开！")
        return
    cap = cv2.VideoCapture(0)
    # 固定分辨率，避免尺寸波动导致闪烁
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # 同步快门（50Hz环境设为~20ms快门，需根据设备调试，-8为示例值）
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)
    # 启用自动曝光（但限制增益波动范围）
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75表示半手动模式，允许一定自动调整
    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头！")
        return
    running = True
    # 启动子线程更新画面（避免阻塞GUI）
    threading.Thread(target=update_camera, daemon=True).start()

    # ✅ 显示右侧预览区
    preview_frame.pack(side=tk.RIGHT, padx=(0, 10))

    # ✅ 打开相机时，加载最近四张
    load_last_photos_from_disk()

    threading.Thread(target=update_camera, daemon=True).start()


def update_camera():
    """相机线程（含剪刀手持续2 s触发）"""
    global cap, running, current_frame, last_capture_time, frame_buffer, \
           last_v_sign_time, v_sign_duration   # 新增全局变量

    while running:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # 帧缓冲
            frame_buffer.append(frame.copy())
            if len(frame_buffer) < 2:
                continue
            frame = frame_buffer.popleft()
            # ===== 镜像 =====
            frame = cv2.flip(frame, 1)

            # 弱光增强（原逻辑）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness < 60:
                frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_result = face_mesh.process(rgb)
            hand_result = hands.process(rgb)
            face_detect_result = face_detect.process(rgb)

            frame = beautify(
                frame,
                face_result,
                face_detect_result.detections if face_detect_result else None,
                beauty_enabled
            )

            # ===== 剪刀手识别（时间累积，更灵敏）=====
            v_sign_flag = False

            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                # 只要有一只手是 V
                if is_v_sign(hand_result):
                    v_sign_flag = True

            # —— 关键：帧累计逻辑（就在这里）——
            if v_sign_flag:
                v_sign_frames += 1
            else:
                v_sign_frames = 0

            # 显示调试信息（可留）
            cv2.putText(
                frame,
                f"V-SIGN FRAMES: {v_sign_frames}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # —— 触发拍照（约 0.5 秒即可）——
            now = time.time()
            if v_sign_frames >= 10 and now - last_capture_time > 2:
                capture_photo(auto=True)
                last_capture_time = now
                v_sign_frames = 0

            # ===== 笑脸触发（原逻辑） =====
            if is_smiling(face_result):
                if smile_start_time == 0:
                    smile_start_time = now
                elif now - smile_start_time >= 0.5 and now - last_capture_time > 2:
                    capture_photo(auto=True)
                    last_capture_time = now
                    smile_start_time = 0
            else:
                smile_start_time = 0

            # 画人脸框（原逻辑）
            if face_detect_result.detections:
                for detection in face_detect_result.detections:
                    mp_draw.draw_detection(frame, detection)

            current_frame = frame

            # 显示到 Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((960, 540))
            imgtk = ImageTk.PhotoImage(image=img)
            root.after(0, lbl_video.update_image, imgtk)

            time.sleep(0.05)
        except Exception as e:
            print(f"帧处理异常：{e}")
            time.sleep(0.1)

def take_photo():
    """手动拍照（保持原逻辑）"""
    capture_photo(auto=False)


def close_camera():
    """关闭相机（保持原逻辑）"""
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    lbl_video.configure(image='')
    # ✅ 隐藏右侧预览框
    preview_frame.pack_forget()
    messagebox.showinfo("退出", "相机已关闭。")


# ---------------- GUI界面 ----------------
root = tk.Tk()
root.title("BeautyCam")
root.geometry("1250x700")

# 使用自定义双缓冲标签（替换原tk.Label）
main_frame = tk.Frame(root)
main_frame.pack(padx=20, pady=10)

video_frame = tk.Frame(main_frame)
video_frame.pack(side=tk.LEFT, padx=(0, 20))

lbl_video = DoubleBufferLabel(video_frame)
lbl_video.pack()

preview_frame = tk.Frame(
    main_frame,
    width=260,
    height=520,
    bg="#1e1e1e"
)
preview_frame.pack_propagate(False)

# ❗ 注意：这里先不 pack


preview_labels = []

for i in range(4):
    lbl = tk.Label(
        preview_frame,
        bg="black",
        cursor="hand2"
    )
    lbl.pack(pady=6)

    lbl.bind(
        "<Button-1>",
        lambda e, idx=i: (
            show_large_image(recent_photos[idx])
            if idx < len(recent_photos) else None
        )
    )

    preview_labels.append(lbl)



# 按钮区（保持原布局）
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

btn_open = tk.Button(
    frame_buttons,
    text="打开相机",
    command=open_camera,
    width=12,
    height=2
)
btn_open.grid(row=0, column=0, padx=10)

btn_capture = tk.Button(
    frame_buttons,
    text="拍照",
    command=take_photo,
    width=12,
    height=2
)
btn_capture.grid(row=0, column=1, padx=10)

btn_album = tk.Button(
    frame_buttons,
    text="打开相册",
    command=open_album,
    width=12,
    height=2
)
btn_album.grid(row=0, column=2, padx=10)

btn_exit = tk.Button(
    frame_buttons,
    text="退出相机",
    command=close_camera,
    width=12,
    height=2
)
btn_exit.grid(row=0, column=3, padx=10)

btn_beauty = tk.Button(
    frame_buttons,
    text="关闭美颜",
    command=toggle_beauty,
    width=12,
    height=2
)
btn_beauty.grid(row=1, column=1, padx=10, pady=10)

btn_eye = tk.Button(
    frame_buttons,
    text="开启大眼",
    command=toggle_eye,
    width=12,
    height=2
)
btn_eye.grid(row=1, column=2, padx=10, pady=10)

# 主循环
root.mainloop()

# 资源释放（程序退出时）
face_mesh.close()
hands.close()
face_detect.close()
import customtkinter as ctk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
from collections import deque, Counter
import threading
import sounddevice as sd
import numpy as np
import time

# ---------------- SETTINGS ----------------
MODEL = YOLO("yolov8s.pt") 
CONF = 0.45 
WINDOW = 15

# Strict Classroom & Student Objects
ALLOWED = [
    "person", "backpack", "handbag", "suitcase", "tie", 
    "laptop", "cell phone", "chair", "dining table", "tv", 
    "mouse", "keyboard", "book", "clock", "scissors", 
    "bottle", "cup", "apple", "sandwich", "sports ball"
]

# THE TRANSLATOR: Forcing YOLO to speak "Classroom"
LABEL_FIX = {
    "backpack": "Bag / Backpack",
    "handbag": "Bag",
    "suitcase": "Teacher Bag",
    "dining table": "Student Desk",
    "tv": "Smartboard",
    "cell phone": "Phone",
    "book": "Book / Notebook",
    "bottle": "Water Bottle",
    "cup": "Drink",
    "tie": "Teacher's Tie",
    "sports ball": "Distraction / Ball"
}

activity_history = deque(maxlen=WINDOW)
event_log = deque(maxlen=8)

cap = None
running = False
latest_frame = None
voice_detected = False

# Threading Variables
ai_frame = None
ai_results_cache = []
current_activity_text = "Classroom Empty"

# ---------------- CLASSROOM BEHAVIOR ENGINE ----------------
def predict_activity(labels):
    # Convert labels to lowercase for easier logic matching
    L = set([lbl.lower() for lbl in labels])
    
    if "person" not in L: return ["Classroom Empty"]
    
    activities = []
    
    # 1. Study & Work
    if {"laptop", "keyboard"} & L or {"laptop", "mouse"} & L: 
        activities.append("Typing / Computer Work")
    if {"book / notebook", "student desk"} <= L or {"book / notebook"} & L: 
        activities.append("Reading / Writing")
    if {"smartboard"} & L: 
        activities.append("Looking at Board / Presenting")
        
    # 2. Rule Breaking / Distractions
    if {"phone"} & L: 
        activities.append("Distracted: Phone Out")
    if {"distraction / ball"} & L: 
        activities.append("Distracted: Playing")
        
    # 3. Transitions
    if {"bag / backpack", "bag"} & L: 
        activities.append("Packing up / Arriving")
    if {"clock"} & L: 
        activities.append("Checking Time")
        
    # 4. Teacher Presence
    if {"teacher's tie", "teacher bag"} & L: 
        activities.append("Instructor at Desk")
        
    # 5. Basic States
    if {"water bottle", "drink", "apple", "sandwich"} & L: 
        activities.append("Snack / Hydration Break")
    if {"chair", "student desk"} & L and not activities: 
        activities.append("Sitting quietly")
            
    return activities if activities else ["Students Active"]

def voice_listener():
    global voice_detected
    def callback(indata, frames, time_, status):
        global voice_detected
        volume = np.linalg.norm(indata) * 10
        voice_detected = volume > 0.1 

    try:
        with sd.InputStream(callback=callback, channels=1, samplerate=16000):
            while running: sd.sleep(100)
    except:
        voice_detected = False

# --- Background AI Worker ---
def process_ai():
    global ai_results_cache, current_activity_text, activity_history, event_log
    
    while running:
        if ai_frame is not None:
            frame_to_process = ai_frame.copy() 
            results = MODEL(frame_to_process, conf=CONF, verbose=False)
            
            temp_boxes = []
            temp_labels_display = [] 
            
            if results:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    original_label = MODEL.names[cls]
                    
                    if original_label not in ALLOWED: continue
                    
                    # Apply our Classroom Translator
                    display_label = LABEL_FIX.get(original_label, original_label.title())
                    temp_labels_display.append(display_label)
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Blue for people, Orange for objects
                    color = (255, 59, 48) if display_label.lower() == "person" else (0, 122, 255) 
                    temp_boxes.append((display_label, x1, y1, x2, y2, color))
            
            ai_results_cache = temp_boxes
            
            # Run behavior logic
            current_acts = predict_activity(temp_labels_display)
            activity_history.append(", ".join(current_acts))
            
            stable = Counter(activity_history).most_common(1)[0][0] if len(activity_history) >= 8 else ",".join(current_acts)
            
            if voice_detected and "Person" in stable: stable += " + Talking"
            
            current_activity_text = stable

            if not event_log or event_log[-1] != stable:
                if stable != "Classroom Empty": event_log.append(stable)
        
        time.sleep(0.03) 

def process_video():
    global running, latest_frame, cap, ai_frame

    while running and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        ai_frame = frame.copy()

        for label, x1, y1, x2, y2, color in ai_results_cache:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            l_x1, l_y1 = max(0, x1), max(0, y1 - 25)
            l_x2, l_y2 = min(frame.shape[1], x1 + w + 10), min(frame.shape[0], y1)
            
            lbl_roi = frame[l_y1:l_y2, l_x1:l_x2]
            if lbl_roi.size != 0:
                blurred_lbl = cv2.GaussianBlur(lbl_roi, (21, 21), 0)
                white_lbl = np.full(lbl_roi.shape, 255, dtype=np.uint8)
                frosted_lbl = cv2.addWeighted(blurred_lbl, 0.6, white_lbl, 0.4, 0)
                frame[l_y1:l_y2, l_x1:l_x2] = frosted_lbl
                cv2.rectangle(frame, (l_x1, l_y1), (l_x2, l_y2), (230, 230, 230), 1)
                
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

        hx1, hy1, hx2, hy2 = 15, 15, 600, 100
        hud_roi = frame[hy1:hy2, hx1:hx2]
        
        if hud_roi.size != 0:
            blurred_hud = cv2.GaussianBlur(hud_roi, (45, 45), 0)
            white_hud = np.full(hud_roi.shape, 255, dtype=np.uint8)
            frosted_hud = cv2.addWeighted(blurred_hud, 0.5, white_hud, 0.5, 0)
            frame[hy1:hy2, hx1:hx2] = frosted_hud
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (240, 240, 240), 2) 

        cv2.putText(frame, f"CLASS STATUS: {current_activity_text}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (30, 30, 32), 1, cv2.LINE_AA)
        
        v_text, v_color = ("AUDIO: CLASS NOISE DETECTED", (0, 122, 255)) if voice_detected else ("AUDIO: SILENT", (142, 142, 147))
        cv2.putText(frame, v_text, (30, 85), cv2.FONT_HERSHEY_DUPLEX, 0.6, v_color, 1, cv2.LINE_AA)

        p_w, p_h = video_label.winfo_width(), video_label.winfo_height()
        if p_w > 100 and p_h > 100:
            target_w = p_w
            target_h = int(target_w * 9 / 16)
            if target_h > p_h:
                target_h = p_h
                target_w = int(target_h * 16 / 9)
            frame = cv2.resize(frame, (target_w, target_h))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        latest_frame = ImageTk.PhotoImage(Image.fromarray(img))
        
    stop_video()

# ---------------- UI CONTROLS ----------------
def start_camera():
    global cap, running
    stop_video()
    cap = cv2.VideoCapture(0)
    running = True
    
    threading.Thread(target=process_video, daemon=True).start()
    threading.Thread(target=process_ai, daemon=True).start()
    threading.Thread(target=voice_listener, daemon=True).start()

def upload_video():
    global cap, running
    stop_video()
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if path:
        cap = cv2.VideoCapture(path)
        running = True
        threading.Thread(target=process_video, daemon=True).start()
        threading.Thread(target=process_ai, daemon=True).start()
        threading.Thread(target=voice_listener, daemon=True).start()

def stop_video():
    global running, cap, latest_frame, ai_results_cache, current_activity_text
    running = False
    if cap:
        cap.release()
        cap = None
    latest_frame = None
    ai_results_cache = []
    current_activity_text = "Classroom Empty"
    video_label.configure(image=None, text="Camera Standby")

def quit_app():
    stop_video()
    app.destroy()

def update_ui():
    if latest_frame:
        video_label.configure(image=latest_frame, text="")
        video_label.imgtk = latest_frame
    app.after(15, update_ui)

def update_log():
    log_box.configure(state="normal")
    log_box.delete("1.0", "end")
    for e in list(event_log):
        log_box.insert("end", f"􀌗  {e}\n") 
    log_box.configure(state="disabled")
    app.after(500, update_log)

# ---------------- APPLE STYLE MAIN WINDOW ----------------
ctk.set_appearance_mode("light") 

app = ctk.CTk()
app.title("EduVision AI - Classroom Edition")
app.geometry("1100x700")
app.minsize(900, 600)
app.configure(fg_color="#F2F2F7") 

app.bind("<Escape>", lambda e: quit_app()) 

app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=0)
app.grid_rowconfigure(0, weight=1)

# Main Video Area
video_frame = ctk.CTkFrame(app, fg_color="#FFFFFF", corner_radius=20, border_width=0)
video_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
video_frame.grid_columnconfigure(0, weight=1)
video_frame.grid_rowconfigure(0, weight=1)

video_label = ctk.CTkLabel(video_frame, text="Camera Standby", font=("Helvetica Neue", 20, "bold"), text_color="#8E8E93")
video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

# Right Control Panel
panel = ctk.CTkFrame(app, width=280, fg_color="#FFFFFF", corner_radius=20, border_width=0)
panel.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)

ctk.CTkLabel(panel, text="Teacher Desk", font=("Helvetica Neue", 22, "bold"), text_color="#1C1C1E").pack(pady=(30, 5))
ctk.CTkLabel(panel, text="EduVision AI • v5.0", font=("Helvetica Neue", 13), text_color="#8E8E93").pack(pady=(0, 25))

btn_font = ("Helvetica Neue", 13, "bold") 
ctk.CTkButton(panel, text="Start Live Feed", command=start_camera, font=btn_font, height=32, corner_radius=8, fg_color="#007AFF", hover_color="#0056b3", text_color="#FFFFFF").pack(pady=(10, 5), padx=25, fill="x")
ctk.CTkButton(panel, text="Load Recording", command=upload_video, font=btn_font, height=32, corner_radius=8, fg_color="#E5E5EA", hover_color="#D1D1D6", text_color="#1C1C1E").pack(pady=5, padx=25, fill="x")
ctk.CTkButton(panel, text="Stop Feed", command=stop_video, font=btn_font, height=32, corner_radius=8, fg_color="#FF3B30", hover_color="#C92A20", text_color="#FFFFFF").pack(pady=5, padx=25, fill="x")
ctk.CTkButton(panel, text="Exit Application", command=quit_app, font=btn_font, height=32, corner_radius=8, fg_color="#1C1C1E", hover_color="#000000", text_color="#FFFFFF").pack(pady=(5, 10), padx=25, fill="x")

ctk.CTkLabel(panel, text="Recent Behavior Log", font=("Helvetica Neue", 15, "bold"), text_color="#1C1C1E").pack(pady=(20, 5), anchor="w", padx=25)

log_box = ctk.CTkTextbox(panel, height=220, font=("Helvetica Neue", 14), fg_color="#F2F2F7", text_color="#1C1C1E", border_width=0, corner_radius=12)
log_box.pack(pady=(0, 25), padx=25, fill="both", expand=True)

update_ui()
update_log()
app.mainloop()
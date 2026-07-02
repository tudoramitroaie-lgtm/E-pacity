import threading
import pyaudio
import numpy as np
import wave
import ollama
import json
import pickle
import time
import cv2
from faster_whisper import WhisperModel
from ultralytics import YOLO
from epacity_energy import safe_set_device_state, generate_report
from plug_control import turn_on_plug, turn_off_plug

MIC_RATE = 16000
MIC_CHUNK = 512
RECORD_SECONDS = 5
WAKE_THRESHOLD = 0.97
POSE_MODEL = "yolov8n-pose.pt"
DEVICE_MODEL = "/home/nvidia/models/fan_detector.pt"
POINT_HOLD_SECONDS = 2.0
CONFIDENCE_THRESHOLD = 0.5
DEVICE_CONFIDENCE = 0.55
FAN_ZONE_LEFT = 400
FAN_ZONE_RIGHT = 640
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 6, 8, 10
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 5, 7, 9

device_lock = threading.Lock()

def turn_on(device, reason="manual"):
    with device_lock:
        success = safe_set_device_state(device, "on", reason=reason)
        if success:
            turn_on_plug()
            print(f"E-pacity: {device.upper()} turned ON (reason)")

def turn_off(device, reason="manual"):
    with device_lock:
        success = safe_set_device_state(device, "off", reason=reason)
        if success:
            turn_off_plug()
            print(f"E-pacity: {device.upper()} turned OFF (reason)")

def toggle(device, reason="manual"):
    with device_lock:
        from epacity_energy import load_log
        log = load_log()
        if device in log and log[device]["state"] == "on":
            safe_set_device_state(device, "off", reason=reason)
            turn_off_plug()
            print(f"E-pacity: {device.upper()} turned OFF via gesture")
        else:
            safe_set_device_state(device, "on", reason=reason)
            turn_on_plug()
            print(f"E-pacity: {device.upper()} turned ON via gesture")

def voice_thread_function():
    print("[Voice] Loading Whisper model...")
    whisper = WhisperModel("tiny")
    print("[Voice] Loading wake word model...")
    with open("/home/nvidia/epacity_model.pkl", "rb") as f:
        wake_model = pickle.load(f)
    print("[Voice] Ready! Say E-pacity to wake me up...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=MIC_RATE,
                    input=True, input_device_index=0, frames_per_buffer=MIC_CHUNK)

    def extract_features(audio):
        audio = audio.astype(np.float32) / 32768.0
        features = []
        for i in range(0, len(audio) - MIC_CHUNK, MIC_CHUNK):
            chunk = audio[i:i + MIC_CHUNK]
            features.extend([np.mean(chunk), np.std(chunk),
                            np.max(np.abs(chunk)), np.mean(np.abs(chunk))])
        features = features[:200]
        while len(features) < 200:
            features.append(0.0)
        return features

    buffer = []
    BUFFER_SIZE = 50
    while True:
        audio_chunk = stream.read(MIC_CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        buffer.append(audio_array)
        if len(buffer) > BUFFER_SIZE:
            buffer.pop(0)
        if len(buffer) == BUFFER_SIZE:
            combined = np.concatenate(buffer)
            features = extract_features(combined)
            prob = wake_model.predict_proba([features])[0][1]
            if prob > WAKE_THRESHOLD:
                print("\n[Voice] Wake word detected! Listening for command...")
                buffer = []
                frames = []
                for _ in range(0, int(MIC_RATE / MIC_CHUNK * RECORD_SECONDS)):
                    data = stream.read(MIC_CHUNK, exception_on_overflow=False)
                    frames.append(data)
                with wave.open("/home/nvidia/command.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(MIC_RATE)
                    wf.writeframes(b"".join(frames))
                segments, _ = whisper.transcribe("/home/nvidia/command.wav")
                command_text = " ".join([s.text for s in segments])
                print(f"[Voice] You said: {command_text}")
                response = ollama.chat(model="llama3.2:1b", messages=[
                    {"role": "system", "content": """You are E-pacity, a smart home assistant. Parse voice commands and return ONLY a JSON object.
Devices: "fan"
Actions: "turn_on", "turn_off", "toggle"
Example: {"device": "fan", "action": "turn_on"}
If unclear return: {"device": "unknown", "action": "unknown"}"""},
                    {"role": "user", "content": command_text}
                ])
                try:
                    raw = response["message"]["content"]
                    # Extract JSON even if there is extra text
                    import re
                    match = re.search(r"\{.*\}", raw, re.DOTALL)
                    result = json.loads(match.group()) if match else {"device": "unknown", "action": "unknown"}
                    parsed_device = result["device"]
                    parsed_action = result["action"]
                    if parsed_device != "unknown":
                        if parsed_action == "turn_on":
                            turn_on(parsed_device, reason="manual")
                        elif parsed_action == "turn_off":
                            turn_off(parsed_device, reason="manual")
                        elif parsed_action == "toggle":
                            toggle(parsed_device, reason="manual")
                    else:
                        print("[Voice] E-pacity: Sorry, I did not understand that command.")
                except Exception as e:
                    import traceback
                    print(f"[Voice] E-pacity: Could not parse command. Error: {e}")
                    print(f"[Voice] Raw response was: {raw}")
                    traceback.print_exc()
                print("\n[Voice] Listening for wake word...")

def gesture_thread_function():
    print("[Gesture] Loading YOLOv8-Pose model...")
    pose_model = YOLO(POSE_MODEL)
    print("[Gesture] Loading fan detection model...")
    device_model = YOLO(DEVICE_MODEL)
    print("[Gesture] Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/nvidia/epacity_recording.mp4', fourcc, fps, (width, height))
    print("[Gesture] Ready! Point at your fan to toggle it.")

    def extend_ray(elbow, wrist, length=2000):
        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            return wrist
        dx, dy = dx / norm, dy / norm
        return (int(wrist[0] + dx * length), int(wrist[1] + dy * length))

    def line_intersects_box(p1, p2, box):
        x1, y1, x2, y2 = box
        for i in range(101):
            t = i / 100
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    pointing_start_time = {}
    triggered_recently = {}
    
    # Presence detection variables
    last_seen_time = time.time()
    ABSENCE_THRESHOLD = 10.0  # seconds before turning off
    was_absent = False
    state_before_absence = "off"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pose_results = pose_model(frame, verbose=False)
        device_results = device_model(frame, conf=DEVICE_CONFIDENCE, verbose=False)
        detected_devices = {}
        for box in device_results[0].boxes:
            label = device_results[0].names[int(box.cls)]
            coords = box.xyxy[0].tolist()
            conf = float(box.conf)
            x_center = (coords[0] + coords[2]) / 2
            if True:  # Detect fan anywhere in frame
                detected_devices["fan"] = coords
                x1, y1, x2, y2 = [int(c) for c in coords]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Fan {conf:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        currently_pointing_at = None
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints_data = pose_results[0].keypoints.data
            for person_kpts in keypoints_data:
                kpts = person_kpts.cpu().numpy()
                for s_idx, e_idx, w_idx in [(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
                                              (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)]:
                    s, e, w = kpts[s_idx], kpts[e_idx], kpts[w_idx]
                    if e[2] < CONFIDENCE_THRESHOLD or w[2] < CONFIDENCE_THRESHOLD:
                        continue
                    elbow_pt = (int(e[0]), int(e[1]))
                    wrist_pt = (int(w[0]), int(w[1]))
                    ray_end = extend_ray(elbow_pt, wrist_pt)
                    if s[2] > CONFIDENCE_THRESHOLD:
                        cv2.line(frame, (int(s[0]), int(s[1])), elbow_pt, (255, 255, 0), 3)
                    cv2.line(frame, elbow_pt, wrist_pt, (255, 255, 0), 3)
                    cv2.line(frame, wrist_pt, ray_end, (0, 255, 0), 2)
                    cv2.circle(frame, wrist_pt, 6, (0, 0, 255), -1)
                    for device_label, coords in detected_devices.items():
                        if line_intersects_box(wrist_pt, ray_end, coords):
                            currently_pointing_at = device_label
                            break
                    break
        now = time.time()
        
        # Presence detection
        person_detected = len(pose_results) > 0 and pose_results[0].keypoints is not None and len(pose_results[0].keypoints.data) > 0
        
        if person_detected:
            last_seen_time = now
            if was_absent:
                # Person returned - restore previous state
                was_absent = False
                print("[Presence] Person detected! Restoring previous state...")
                from epacity_energy import load_log
                log = load_log()
                if state_before_absence == "on":
                    safe_set_device_state("fan", "on", reason="auto")
                    turn_on_plug()
                    print("[Presence] Fan restored to ON")
                else:
                    print("[Presence] Fan staying OFF (was off before you left)")
        else:
            absence_duration = now - last_seen_time
            if absence_duration >= ABSENCE_THRESHOLD and not was_absent:
                # Person left - save state and turn off
                from epacity_energy import load_log
                log = load_log()
                state_before_absence = log.get("fan", {}).get("state", "off")
                was_absent = True
                print(f"[Presence] No person detected for {ABSENCE_THRESHOLD}s - turning fan off (was {state_before_absence})")
                safe_set_device_state("fan", "off", reason="auto")
                turn_off_plug()

        if currently_pointing_at:
            if currently_pointing_at not in pointing_start_time:
                pointing_start_time[currently_pointing_at] = now
            held_duration = now - pointing_start_time[currently_pointing_at]
            progress = min(held_duration / POINT_HOLD_SECONDS, 1.0)
            if currently_pointing_at in detected_devices:
                x1, y1, x2, y2 = [int(c) for c in detected_devices[currently_pointing_at]]
                cv2.putText(frame, f"Pointing: {progress*100:.0f}%", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if held_duration >= POINT_HOLD_SECONDS:
                last_trigger = triggered_recently.get(currently_pointing_at, 0)
                if now - last_trigger > 5.0:
                    toggle(currently_pointing_at, reason="manual")
                    triggered_recently[currently_pointing_at] = now
                pointing_start_time[currently_pointing_at] = now
        else:
            pointing_start_time = {}
        cv2.imshow("E-pacity", frame)
        key = cv2.waitKey(1) & 0xFF
        out.write(frame)
        if key == ord('q'):
            print("\n[Gesture] Q pressed - saving video and stopping...")
            break
        time.sleep(0.05)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Video saved to epacity_recording.avi")

if __name__ == "__main__":
    print("=" * 50)
    print("  E-PACITY - Final System")
    print("  Voice + AI Gesture + Smart Plug + Energy Logger")
    print("=" * 50)
    voice_thread = threading.Thread(target=voice_thread_function, daemon=True)
    gesture_thread = threading.Thread(target=gesture_thread_function, daemon=True)
    voice_thread.start()
    gesture_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down E-pacity...")
        generate_report()
        print("Demo video saved to epacity_demo_final.mp4")

# E-pacity 

**Your room's full potential, intelligently managed.**

An offline edge AI system running entirely on a NVIDIA Jetson Orin Nano that makes your room intelligently aware of you. No cloud. All privacy.

---

## Demo Video

> *Add your demo video link here*

---

## What It Does

E-pacity lets you control smart devices in your room through three natural interaction methods:

- **Voice control** — say "E-pacity, turn on the fan" and it responds
- **Gesture control** — point at a device for 2 seconds to toggle it
- **Presence detection** — automatically turns devices off when you leave the room and restores them when you return
- **Energy tracking** — logs how much energy E-pacity saves by managing devices automatically

Everything runs locally on the Jetson — no internet connection required, no data sent to any server.

---

## System Architecture

```
[ Logitech C270 Webcam + Mic ]
              │
              ▼
    [ NVIDIA Jetson Orin Nano ]
              │
    ┌─────────┼──────────────────┐
    │         │                  │
    ▼         ▼                  ▼
[ Voice ]  [ Gesture +       [ Energy
Pipeline]   Presence ]        Logger ]
    │         │                  │
    ▼         ▼                  │
Whisper   YOLOv8-Pose            │
(speech)  + Custom Fan           │
    │       Detector             │
    ▼         │                  │
Llama 3.2    Ray                 │
(parsing)   Intersection         │
    │         │                  │
    └────┬────┘                  │
         ▼                       │
   Device Command ───────────────┘
         │
         ▼
   Tapo Smart Plug (local Wi-Fi)
         │
         ▼
        Fan
```

---

## Technology Stack

| Component | Technology | Category |
|---|---|---|
| Speech-to-text | Faster-Whisper (tiny model) | Deep Learning |
| Command understanding | Llama 3.2 1B (via Ollama) | Deep Learning / LLM |
| Wake word detection | Custom MLPClassifier (scikit-learn) | Machine Learning |
| Pose estimation | YOLOv8-Pose (Ultralytics) | Deep Learning / Computer Vision |
| Device detection | Custom YOLOv8-Nano (trained on fan) | Deep Learning / Computer Vision |
| Presence detection | YOLOv8-Pose person detection | Computer Vision |
| Device control | Tapo Python library (local Wi-Fi) | Networking |
| Energy tracking | Custom Python state logger | Software Engineering |

All AI inference and device control runs locally on the Jetson Orin Nano's GPU. No cloud services are used.

---

## Hardware Requirements

- NVIDIA Jetson Orin Nano
- Logitech C270 HD Webcam
- TP-Link Tapo TP15 Smart Plug (or compatible)

---

## Installation

### Step 1 — Install Ollama and pull the language model
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

### Step 2 — Install Python dependencies
```bash
sudo pip3 install faster-whisper ultralytics opencv-python pyaudio ollama
sudo pip3 install scikit-learn numpy tapo python-dotenv
sudo apt install -y python3-pyaudio portaudio19-dev ffmpeg
```

### Step 3 — Set up environment variables
Create a `.env` file in the project root:
```
TAPO_EMAIL=your_tapo_email@gmail.com
TAPO_PASSWORD=your_tapo_password
TAPO_PLUG_IP=your_plug_ip_address
```

To find your plug's IP address, open the Tapo app → tap your device → gear icon → Device Info.

### Step 4 — Train your own wake word model
Record 50 samples of yourself saying "E-pacity":
```bash
python3 record_samples.py
```
Then train the classifier:
```bash
python3 train_epacity.py
```

### Step 5 — Train your own device detector
Take photos of your device using the capture tool in NoMachine:
```bash
export DISPLAY=:0
python3 capture_photos.py
```
Upload photos to [Roboflow](https://roboflow.com), label with bounding boxes, export as YOLOv8 format, download and transfer to Jetson, then train:
```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='your_dataset/data.yaml', epochs=30, imgsz=640, device=0)
print('Done!')
"
```
Update `DEVICE_MODEL` path in `epacity_final.py` to point to your trained model weights.

### Step 6 — Connect your smart plug
1. Plug the Tapo TP15 into a wall outlet
2. Open the Tapo app and connect it to your Wi-Fi network
3. Make sure your Jetson is on the same Wi-Fi network
4. Add the plug's IP address to your `.env` file

### Step 7 — Run E-pacity
```bash
ollama serve &
export DISPLAY=:0
python3 epacity_final.py
```

---

## How It Works

### Voice Pipeline
The C270's built-in microphone continuously streams audio to a custom-trained wake word model. This model was trained from scratch using 50 voice recordings of "E-pacity" recorded directly on the Jetson and trained using scikit-learn's MLPClassifier — making the wake word unique to this project and trained on the user's own voice.

Once the wake word is detected, the system records a 5-second audio clip and transcribes it locally using Faster-Whisper. The transcribed text is passed to Llama 3.2 1B running locally via Ollama, which returns structured JSON identifying the device and action. A keyword fallback system ensures reliability even when speech recognition isn't perfect.

### Gesture Pipeline
YOLOv8-Pose processes each camera frame to extract body keypoints (shoulder, elbow, wrist). A geometric ray is calculated extending from the elbow through the wrist across the frame. Simultaneously, a custom-trained YOLOv8-Nano model detects the fan and draws a bounding box around it in real time. If the pointing ray intersects the detected device box for 2 sustained seconds, the corresponding device is toggled — preventing accidental triggers from normal arm movement.

### Presence Detection
YOLOv8-Pose continuously monitors whether a person is visible in the camera frame. If no person is detected for 10 consecutive seconds, E-pacity saves the current device state and turns everything off automatically (logged as an energy saving event). When the person returns to the frame, E-pacity restores the previous state — so if the fan was on when you left, it turns back on when you return.

### Energy Logger
Every device state change is recorded with a timestamp and a reason — `manual` (user-initiated via voice or gesture) or `auto` (E-pacity-initiated via presence detection). Only auto-initiated off events are credited as genuine energy savings, since these represent usage that would have otherwise continued unnecessarily. The logger also gracefully handles the smart plug's scheduled offline maintenance window (3-5 AM) to avoid recording inaccurate state data.

---

## Project Name

**E-pacity** combines "E" (electric) with "-pacity" (from capacity) — representing a room's full electric potential, intelligently managed.

---

## Known Limitations

- The custom wake word model occasionally triggers on similar-sounding phrases. A larger and more varied training dataset would improve precision.
- Device detection range is approximately 2 meters. Training with additional photos taken at greater distances would extend this range.
- The system currently supports one device (fan). Additional devices can be added by collecting training photos and updating the device labels.

---

## Future Improvements

- Support for multiple devices simultaneously (lamp, speaker, etc.)
- TensorRT optimization of Whisper and YOLOv8 for faster GPU inference on Jetson
- Custom wake word trained with more samples for higher accuracy
- Time-of-day awareness (different device preferences at different times)
- Mobile app for energy report viewing

---

## File Structure

```
E-pacity/
├── epacity_final.py      # Main system script (voice + gesture + presence)
├── epacity_energy.py     # Energy logger
├── plug_control.py       # Smart plug control via local Wi-Fi
├── record_samples.py     # Wake word sample recorder
├── train_epacity.py      # Wake word model trainer
├── capture_photos.py     # Device photo capture tool
├── epacity_model.pkl     # Trained wake word classifier
├── .env                  # Tapo credentials (not included in repo)
├── .env.example          # Example environment file
├── .gitignore            # Excludes .env from version control
└── README.md             # This file
```

---

## Resources

- [Ultralytics YOLOv8 Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [Faster-Whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Ollama Python Client](https://github.com/ollama/ollama-python)
- [Jetson-inference by dusty-nv](https://github.com/dusty-nv/jetson-inference)
- [NVIDIA Jetson Community Projects](https://developer.nvidia.com/embedded/community/jetson-projects)
- [Roboflow - Dataset labeling and training](https://roboflow.com)

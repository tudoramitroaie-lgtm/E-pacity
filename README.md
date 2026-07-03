# E-pacity 

<img width="962" height="596" alt="Screenshot 2026-07-02 144643" src="https://github.com/user-attachments/assets/daa69fef-34b6-4c15-86c2-ba219bd68fd8" />

**Your room's full potential, intelligently managed.**

E-pacity is an intelligent room automation system running entirely on a NVIDIA Jetson Orin Nano. Using a standard webcam and microphone, it lets you control smart devices through voice commands, hand gestures, and automatic presence detection — with no cloud dependency and no external servers involved.

No cloud. All privacy.

---

## Demo

https://drive.google.com/file/d/1tbE74-gjbUoxgAotaAf24MDGpMJprv8L/view?usp=drive_link

---

## What it does

- Say **"E-pacity"**, wait for the confirmation, then give your command naturally
- **Point at a device** and hold for a couple of seconds to toggle it
- **Walk out of the room** and it turns devices off automatically
- **Walk back in** and it restores whatever state you left things in
- At the end of a session it prints an **energy report** showing real usage and savings

> After saying "E-pacity", pause and wait for the wake word confirmation before speaking your command. For gesture control, keep the point steady and give it a couple of seconds to register.

---

## How it works

**Voice**
The microphone listens continuously for the wake word "E-pacity" using a small neural network I trained from scratch on 50 recordings of my own voice. Once triggered, it records the command, transcribes it locally with Faster-Whisper, and sends the text to Llama 3.2 running locally via Ollama. Llama figures out the device and action and returns a structured response that gets executed immediately.

**Gesture**
The camera runs YOLOv8-Pose to track my arm skeleton in real time. It projects a ray from my elbow through my wrist and checks whether that ray crosses the bounding box of the target device — detected by a second YOLOv8 model I trained on 90 photos of the actual fan. Holding the point for 2 seconds triggers the action. The hold requirement prevents accidental triggers from normal arm movement.

**Presence detection**
The same pose model monitors whether anyone is in the frame. If the room appears empty for 10 seconds, E-pacity saves the current device state and turns everything off, logging it as an automatic energy saving. When the user returns, it restores the previous state.

**Energy logger**
Every device change is recorded with a timestamp and a reason — manual for voice or gesture, auto for presence-triggered changes. Only automatic off-events count as savings. At the end of a session it prints a full report.

<img width="2558" height="1532" alt="Screenshot 2026-06-30 100351" src="https://github.com/user-attachments/assets/13fa57a8-d562-4357-94f5-0dc4583969a3" />


---

## Devices used

| Component | Technology | Category |
|---|---|---|
| Wake word | Custom MLPClassifier trained on my voice | Machine Learning |
| Speech-to-text | Faster-Whisper (tiny model) | Deep Learning |
| Command parsing | Llama 3.2 1B via Ollama | Deep Learning / LLM |
| Pose tracking | YOLOv8-Pose (Ultralytics) | Computer Vision |
| Device detection | Custom YOLOv8-Nano (90 training photos) | Deep Learning |
| Presence detection | YOLOv8-Pose person check | Computer Vision |
| Device control | Tapo Python library over local Wi-Fi | Networking |
| Energy tracking | Custom Python logger | Software |

---

## Hardware

- NVIDIA Jetson Orin Nano
- Logitech C270 HD Webcam
- TP-Link Tapo TP15 Smart Plug

---

## Setup

### 1. Install Ollama

Run `curl -fsSL https://ollama.com/install.sh | sh` to install Ollama, then run `ollama pull llama3.2:1b` to download the language model.

<img width="798" height="233" alt="Screenshot 2026-06-29 173938" src="https://github.com/user-attachments/assets/9b8e8652-74f0-45f1-9957-fbe2f8f8d385" />


### 2. Install Python packages

Run the following:

`sudo pip3 install faster-whisper ultralytics opencv-python pyaudio ollama scikit-learn numpy tapo python-dotenv`

`sudo apt install -y python3-pyaudio portaudio19-dev ffmpeg`

### 3. Configure credentials

Create a file called `.env` in the project folder with the following:

```
TAPO_EMAIL=your_email@gmail.com
TAPO_PASSWORD=your_password
TAPO_PLUG_IP=your_plug_ip
```

To find the plug IP: open the Tapo app, tap your device, tap the gear icon, then tap Device Info.

### 4. Train the wake word

Run `python3 record_samples.py` to record 50 samples of yourself saying "E-pacity", then run `python3 train_epacity.py` to train the classifier.

<img width="2558" height="1537" alt="Screenshot 2026-06-29 171851" src="https://github.com/user-attachments/assets/b3d82d24-840d-45c5-bd05-f5c7c808ebe6" />

### 5. Train the device detector

Open a terminal in NoMachine and run `export DISPLAY=:0` followed by `python3 capture_photos.py` to take photos of your device.

Upload the photos to [Roboflow](https://roboflow.com), draw bounding boxes around the device in each photo, and export in YOLOv8 format. Download the dataset to the Jetson, then run:

`python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.train(data='dataset/data.yaml', epochs=30, imgsz=640, device=0)"`

Once training finishes, update the `DEVICE_MODEL` path in `epacity_final.py` to point to the new weights file inside the `runs/detect/` folder.

<img width="2558" height="1535" alt="Screenshot 2026-07-01 095910" src="https://github.com/user-attachments/assets/d6a8a109-2b44-4e87-bd3d-cf7a0c042620" />

### 6. Run it

Start Ollama in the background with `ollama serve &`, then open NoMachine and run:

`export DISPLAY=:0`

`python3 epacity_final.py`

Press **Q** in the camera window to stop the session cleanly.

<img width="2558" height="1598" alt="Screenshot 2026-07-01 101809" src="https://github.com/user-attachments/assets/a497f6d8-ed62-4261-bfdc-e9e05f6a006a" />

---

## File structure

- **epacity_final.py** — main script, runs voice, gesture, and presence detection together
- **epacity_energy.py** — energy logger
- **plug_control.py** — handles plug on/off over local Wi-Fi
- **record_samples.py** — records wake word training samples
- **train_epacity.py** — trains the wake word model
- **capture_photos.py** — photo capture tool for device detector training
- **epacity_model.pkl** — trained wake word classifier
- **.env** — your credentials, not included in the repo
- **.env.example** — template showing what goes in .env


## Known limitations

The wake word occasionally triggers on similar-sounding phrases — more training samples would help. The fan detector works reliably up to about 2 meters; photos taken at greater distances during training would extend this. Currently only one device is supported, but adding more is straightforward — collect photos, retrain, done.


## References

- [YOLOv8 Pose](https://docs.ultralytics.com/tasks/pose/)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Ollama Python](https://github.com/ollama/ollama-python)
- [jetson-inference by dusty-nv](https://github.com/dusty-nv/jetson-inference)
- [Roboflow](https://roboflow.com)
- [NVIDIA Jetson Projects](https://developer.nvidia.com/embedded/community/jetson-projects)

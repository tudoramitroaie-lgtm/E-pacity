import pyaudio
import wave
import os

os.makedirs("wake_word_samples/epacity", exist_ok=True)

p = pyaudio.PyAudio()

for i in range(50):
    input(f"Press Enter to record sample {i+1}/50, then say 'E-pacity'...")
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=0,
                    frames_per_buffer=1280)
    
    print("Recording... say E-pacity now!")
    frames = []
    for _ in range(0, int(16000 / 1280 * 2)):
        data = stream.read(1280)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    filename = f"wake_word_samples/epacity/sample_{i+1}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(frames))
    
    print(f"Saved sample {i+1}")

p.terminate()
print("All samples recorded!")

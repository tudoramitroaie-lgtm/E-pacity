import cv2
import os

# Setup directories
base = "/home/nvidia/jetson-inference/python/training/detection/ssd/data/epacity_devices"
sets = ["train", "val", "test"]
classes = ["Fan", "Lamp"]

for s in sets:
    for c in classes:
        os.makedirs(f"{base}/{s}/{c}", exist_ok=True)

print("Directories created!")
print("Controls:")
print("  F = capture Fan photo")
print("  L = capture Lamp photo")
print("  V = switch to val set")
print("  T = switch to test set")
print("  Q = quit")
print("")

cap = cv2.VideoCapture(0)
current_set = "train"
counts = {"train": {"Fan": 0, "Lamp": 0}, "val": {"Fan": 0, "Lamp": 0}, "test": {"Fan": 0, "Lamp": 0}}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show status on frame
    cv2.putText(frame, f"Set: {current_set}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Fan: {counts[current_set]['Fan']} Lamp: {counts[current_set]['Lamp']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "F=Fan L=Lamp V=val T=test Q=quit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("E-pacity Photo Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('v'):
        current_set = "val"
        print(f"Switched to: val")
    elif key == ord('t'):
        current_set = "test"
        print(f"Switched to: test")
    elif key == ord('f'):
        count = counts[current_set]["Fan"]
        filename = f"{base}/{current_set}/Fan/fan_{count+1:03d}.jpg"
        cv2.imwrite(filename, frame)
        counts[current_set]["Fan"] += 1
        print(f"Saved Fan photo {count+1} ({current_set})")
    elif key == ord('l'):
        count = counts[current_set]["Lamp"]
        filename = f"{base}/{current_set}/Lamp/lamp_{count+1:03d}.jpg"
        cv2.imwrite(filename, frame)
        counts[current_set]["Lamp"] += 1
        print(f"Saved Lamp photo {count+1} ({current_set})")

cap.release()
cv2.destroyAllWindows()

print("\nFinal counts:")
for s in sets:
    for c in classes:
        print(f"  {s}/{c}: {counts[s][c]} photos")

import json
import os
from datetime import datetime, time as dt_time

LOG_FILE = "energy_log.json"

DEVICE_WATTAGE = {
    
    "fan": 35,
}

MAINTENANCE_START = dt_time(3, 0)
MAINTENANCE_END = dt_time(5, 0)

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {
        "light": {"state": "off", "last_change": None, "total_on_minutes": 0, "auto_saved_minutes": 0},
        "speaker": {"state": "off", "last_change": None, "total_on_minutes": 0, "auto_saved_minutes": 0},
        "fan": {"state": "off", "last_change": None, "total_on_minutes": 0, "auto_saved_minutes": 0},
    }

def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def _minutes_since(timestamp_str):
    if timestamp_str is None:
        return 0
    last = datetime.fromisoformat(timestamp_str)
    now = datetime.now()
    return (now - last).total_seconds() / 60

def set_device_state(device, new_state, reason="manual"):
    log = load_log()
    entry = log[device]

    if entry["state"] == "on":
        elapsed = _minutes_since(entry["last_change"])
        entry["total_on_minutes"] += elapsed
        if reason == "auto" and new_state == "off":
            entry["auto_saved_minutes"] += elapsed

    entry["state"] = new_state
    entry["last_change"] = datetime.now().isoformat()
    log[device] = entry
    save_log(log)

    print(f"[Energy Logger] {device.upper()} -> {new_state.upper()} ({reason})")

def is_maintenance_window():
    now = datetime.now().time()
    return MAINTENANCE_START <= now <= MAINTENANCE_END

def safe_set_device_state(device, new_state, reason="manual"):
    if is_maintenance_window():
        print(f"[Energy Logger] Skipping {device} update — plug offline for scheduled maintenance (3-5 AM)")
        return False
    set_device_state(device, new_state, reason)
    return True

def generate_report():
    log = load_log()
    print("\n===== E-PACITY ENERGY REPORT =====")
    total_kwh_saved = 0
    for device, data in log.items():
        watts = DEVICE_WATTAGE.get(device, 10)
        on_hours = data["total_on_minutes"] / 60
        saved_hours = data["auto_saved_minutes"] / 60
        kwh_used = (watts * on_hours) / 1000
        kwh_saved = (watts * saved_hours) / 1000
        total_kwh_saved += kwh_saved

        print(f"\n{device.upper()}:")
        print(f"  Current state:     {data['state'].upper()}")
        print(f"  Total on time:     {data['total_on_minutes']:.1f} min ({on_hours:.2f} hrs)")
        print(f"  Auto-saved time:   {data['auto_saved_minutes']:.1f} min ({saved_hours:.2f} hrs)")
        print(f"  Energy used:       {kwh_used:.4f} kWh")
        print(f"  Energy saved:      {kwh_saved:.4f} kWh")

    print(f"\nTOTAL ENERGY SAVED BY E-PACITY: {total_kwh_saved:.4f} kWh")
    print("===================================\n")
    return total_kwh_saved

if __name__ == "__main__":
    print("Demo: simulating a day of usage...\n")

    set_device_state("light", "on", reason="manual")
    import time
    time.sleep(10)

    set_device_state("light", "off", reason="auto")

    set_device_state("speaker", "on", reason="manual")
    time.sleep(10)

    set_device_state("speaker", "off", reason="manual")

    generate_report()

    print("Testing maintenance window check...")
    print(f"Is it currently maintenance window (3-5 AM)? {is_maintenance_window()}")
    result = safe_set_device_state("light", "on", reason="manual")
    print(f"Command executed successfully: {result}")

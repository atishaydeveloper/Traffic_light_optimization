import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image

# Import your detection and timing functions
from vehicle_detection import load_yolo_model, load_class_names, detect_vehicles, VEHICLE_CLASSES
from green_time_signal import calculate_green_time, assess_congestion, calculate_weighted_density, send_command_to_arduino

# Configuration
BASE_IMAGE_DIR = "images/Intersection1"  # Single intersection folder
DIRECTIONS = ["North", "East", "South", "West"]
MODEL_WEIGHTS = "models/yolov4.weights"
MODEL_CFG = "models/yolov4.cfg"
CLASS_NAMES_FILE = "coco.names"
ARDUINO_PORT = 'COM8'  # Set your Arduino port
ARDUINO_BAUD = 9600

@st.cache_resource
def load_resources():
    net = load_yolo_model(MODEL_WEIGHTS, MODEL_CFG)
    classes = load_class_names(CLASS_NAMES_FILE)
    return net, classes

# Process one direction image, return counts and green time
def process_direction(yolo_net, classes, direction):
    img_path = os.path.join(BASE_IMAGE_DIR, f"{direction}.jpg")
    if not os.path.exists(img_path):
        st.error(f"Image for {direction} not found: {img_path}")
        return None
    image_bgr = cv2.imread(img_path)
    annotated, counts, _ = detect_vehicles(image_bgr, yolo_net, classes, 0.5, 0.4)
    green_time = calculate_green_time(counts)
    density = calculate_weighted_density(counts)
    level, suggestion = assess_congestion(density)
    return {
        "direction": direction,
        "image_bgr": annotated,
        "counts": counts,
        "green_time": green_time,
        "density": density,
        "congestion_level": level,
        "suggestion": suggestion
    }

# Main simulation function
def simulate_intersection(yolo_net, classes):
    st.title("🚦 4‑Direction Intersection Simulation")
    st.write("Analyzing each direction, computing optimized green times, and cycling lights based on density.")

    # Load and process all directions
    data = []
    for d in DIRECTIONS:
        result = process_direction(yolo_net, classes, d)
        if result:
            data.append(result)

    # Sort by descending density to prioritize
    data.sort(key=lambda x: x["density"], reverse=True)

    # Simulate each in turn
    for entry in data:
        dir_name = entry["direction"]
        green_time = entry["green_time"]

        # Send to Arduino: format 'N:15' where N/E/S/W prefix
        prefix = dir_name[0]  # 'N', 'E', 'S', 'W'
        send_command_to_arduino(f"{prefix}:{green_time}", port=ARDUINO_PORT, baud_rate=ARDUINO_BAUD)

        st.header(f"Green Light ➡️ {dir_name}")
        img_rgb = cv2.cvtColor(entry["image_bgr"], cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"{dir_name} view", use_container_width=True)
        cols = st.columns(len(VEHICLE_CLASSES))
        for idx, cls in enumerate(VEHICLE_CLASSES):
            with cols[idx]:
                st.metric(label=cls.capitalize(), value=entry["counts"].get(cls, 0))
        st.metric(label="Calculated Green Time (s)", value=green_time)
        st.markdown(f"**Congestion Level:** {entry['congestion_level']}")
        if entry['suggestion']:
            st.warning(entry['suggestion'])

        st.success(f"🟢 Green ON for {green_time} seconds")
        time.sleep(green_time)
        st.warning("🟡 Yellow Clearance (3s)")
        time.sleep(3)
        st.error("🔴 All Red (2s)")
        time.sleep(2)
        st.write("---")

# Streamlit App
def main():
    yolo_net, classes = load_resources()
    if yolo_net is None or classes is None:
        st.error("Failed to load YOLO model or class names.")
    else:
        if st.button("▶️ Run 4‑Direction Simulation"):
            simulate_intersection(yolo_net, classes)

if __name__ == '__main__':
    main()

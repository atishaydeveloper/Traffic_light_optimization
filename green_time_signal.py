# green_time_signal.py
import numpy as np

# Default time allocation per vehicle type (in seconds)
DEFAULT_TIME_PER_VEHICLE = {
    "car": 1.0,
    "motorbike": 0.5,
    "bus": 3.0,
    "bicycle": 0.5,
    "truck": 2.5
}

# Congestion Thresholds (total weighted vehicle units)
# Adjust these based on intersection capacity and desired sensitivity
CONGESTION_LOW_THRESHOLD = 15
CONGESTION_MEDIUM_THRESHOLD = 35 # Units, not raw count

def calculate_weighted_density(vehicle_counts, weights=DEFAULT_TIME_PER_VEHICLE):
    """Calculates a weighted density score based on vehicle types."""
    density_score = 0
    for vehicle_type, count in vehicle_counts.items():
        density_score += count * weights.get(vehicle_type, 1.0) # Default weight 1 if not found
    return density_score


def calculate_green_time(vehicle_counts, base_time=10, time_per_vehicle=DEFAULT_TIME_PER_VEHICLE, max_time=90):
    """
    Calculates the optimal green light duration based on vehicle counts.

    Args:
        vehicle_counts (dict): Dictionary of detected vehicle counts.
        base_time (int): Minimum green light time (seconds).
        time_per_vehicle (dict): Time addition per vehicle type (seconds).
        max_time (int): Maximum allowed green light time (seconds).

    Returns:
        int: Calculated green light time in seconds.
    """
    calculated_time = float(base_time) # Start with float for precision
    for vehicle_type, count in vehicle_counts.items():
        calculated_time += count * time_per_vehicle.get(vehicle_type, 1.0) # Use get for safety

    # Clamp the time between base_time and max_time
    final_time = int(np.clip(calculated_time, base_time, max_time))
    return final_time

import serial
import time

def send_command_to_arduino(cmd: str, port='COM8', baud_rate=9600):
    """
    Sends an arbitrary command string to Arduino,
    appending a newline, and leaves the port open long enough to complete.
    """
    try:
        with serial.Serial(port, baud_rate, timeout=2) as ser:
            time.sleep(2)  # allow Arduino reset
            full_cmd = cmd.strip() + '\n'
            ser.write(full_cmd.encode())
            print(f"✅ Sent command to Arduino on {port!r}: {full_cmd!r}")
            # wait a bit so Arduino can run its cycle
            # if you know the max expected green time, you could parse seconds from cmd and wait that + buffer
            time.sleep(2)
    except Exception as e:
        print(f"❌ Failed to send data to Arduino: {e}")


def assess_congestion(weighted_density):
    """
    Assesses the congestion level based on weighted density.

    Args:
        weighted_density (float): The calculated weighted density score.

    Returns:
        tuple: A tuple containing:
            - str: Congestion level ("Low", "Medium", "High").
            - str: Traffic redirection suggestion (or empty string).
    """
    if weighted_density < CONGESTION_LOW_THRESHOLD:
        level = "Low"
        suggestion = ""
    elif weighted_density < CONGESTION_MEDIUM_THRESHOLD:
        level = "Medium"
        suggestion = ""
    else:
        level = "High"
        suggestion = "Consider rerouting non-essential traffic."

    return level, suggestion

# Example usage (optional, for testing module directly)
if __name__ == '__main__':
    sample_counts = {"car": 15, "motorbike": 5, "bus": 2, "truck": 1, "bicycle": 3}
    density = calculate_weighted_density(sample_counts)
    green_time = calculate_green_time(sample_counts, base_time=12, max_time=75)
    congestion_level, redirect_suggestion = assess_congestion(density)

    print(f"Sample Counts: {sample_counts}")
    print(f"Weighted Density: {density:.2f}")
    print(f"Calculated Green Time: {green_time} seconds")
    print(f"Congestion Level: {congestion_level}")
    if redirect_suggestion:
        print(f"Suggestion: {redirect_suggestion}")
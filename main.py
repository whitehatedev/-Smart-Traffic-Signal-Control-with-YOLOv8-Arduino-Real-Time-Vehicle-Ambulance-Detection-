import cv2
from ultralytics import YOLO
import numpy as np
import time
import serial  # To communicate with Arduino

# Setup serial communication with Arduino
arduino = serial.Serial('COM8 ', 9600)  # Replace 'COM3' with your Arduino's COM port
time.sleep(2)  # Wait for Arduino to initialize

# Load YOLOv8 model
model = YOLO('a.pt')

# List of class names
class_names = [
    'ambulance', 'Barricade', 'Bike', 'Board', 'Building', 'Bullock-cart', 'Bus',
    'Car', 'Cattle', 'Crane', 'Cycle', 'Dog', 'Electricity-Pole', 'Goat',
    'Lamp-Post', 'Manhole', 'Person', 'Rikshaw', 'Road-Divider', 'Tempo',
    'Tractor', 'Traffic-Sign-Board', 'Traffic-Signal', 'Tree', 'Truck', 'Zebra-Crossing'
]

# Define vehicle-related classes for density counting
vehicle_classes = [
    'ambulance', 'Bike', 'Bus', 'Car', 'Cycle', 'Rikshaw', 'Tempo', 'Tractor', 'Truck','Barricade'
]

# Signal states
signal_states = {
    'North': 'Red',
    'East': 'Red',
    'South': 'Red',
    'West': 'Red'
}

# Signal timing parameters
base_signal_duration = 10  # Default green signal time in seconds
min_signal_duration = 5  # Minimum green signal time
max_signal_duration = 20  # Maximum green signal time
density_scale_factor = 1  # Scale factor for density-based time adjustment

# Initialize signal lamp timers
last_signal_change = time.time()
current_direction_index = 0
directions = ['North', 'East', 'South', 'West']

# Remaining time for current signal
remaining_time = base_signal_duration

# Priority mode
ambulance_detected = False
ambulance_direction = None


def detect_objects(frame):
    """Detect objects in the frame using YOLOv8 and return detected object names."""
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # Class IDs

    for box, confidence, cls in zip(detections, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        confidence_percentage = int(confidence * 100)  # Convert to percentage
        class_name = class_names[int(cls)]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label and confidence percentage
        label = f"{class_name} {confidence_percentage}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Collect detected object names
    detected_objects = [class_names[int(cls)] for cls in classes]
    return detected_objects


def count_objects(objects, target_classes):
    """Count the number of vehicles detected from target classes."""
    return sum(obj in target_classes for obj in objects)


def calculate_signal_time(density):
    """Calculate green signal time based on vehicle density."""
    if density == 0:
        return 0  # Skip signal if density is zero (move to next direction)

    # Scale the density and clamp between min and max durations
    adjusted_time = base_signal_duration + (density * density_scale_factor)

    # If the density is low, decrease the time duration
    if density < 5:  # Assuming 5 is a threshold for low density
        adjusted_time -= 2  # Decrease time for low vehicle density
    elif density > 10:  # Assuming 10 is a threshold for high density
        adjusted_time += 3  # Increase time for high density

    # Clamp to the minimum and maximum allowed durations
    return max(min_signal_duration, min(max_signal_duration, int(adjusted_time)))


def send_signal_state_to_arduino(direction, state):
    """Send the traffic signal state to the Arduino to control LEDs."""
    signal_state = f"{direction}:{state}\n"
    arduino.write(signal_state.encode())  # Send signal state to Arduino


def update_signal_lamps(density, detected_objects):
    """Update traffic signal lamps based on vehicle density and priority mode."""
    global signal_states, last_signal_change, current_direction_index, remaining_time, ambulance_detected, ambulance_direction

    current_time = time.time()

    # Check for ambulance in any direction
    for direction, objects in zip(directions, detected_objects):
        if 'ambulance' in objects:
            ambulance_detected = True
            ambulance_direction = direction
            break
    else:
        ambulance_detected = False  # Reset if no ambulance is detected

    # If an ambulance is detected, give it priority
    if ambulance_detected:
        # Set all signals to Red
        for direction in signal_states:
            signal_states[direction] = 'Red'

        # Set the ambulance direction to Green
        signal_states[ambulance_direction] = 'Green'

        # Notify Arduino of the priority change
        send_signal_state_to_arduino(ambulance_direction, 'Green')

        # Keep the ambulance signal green until it passes
        remaining_time = max(0, int(base_signal_duration - (current_time - last_signal_change)))
        if remaining_time <= 0:
            # Reset priority mode once time expires
            ambulance_detected = False
            last_signal_change = current_time
            current_direction_index = (current_direction_index + 1) % len(directions)
        return remaining_time

    # Normal operation based on vehicle density
    current_direction = directions[current_direction_index]
    num_vehicles = density[current_direction]

    # If the density is 0, skip this direction and move to the next
    if num_vehicles == 0:
        current_direction_index = (current_direction_index + 1) % len(directions)
        remaining_time = 0
        return remaining_time

    # Calculate green signal time based on density
    green_duration = calculate_signal_time(num_vehicles)

    # Remaining time
    remaining_time = max(0, int(green_duration - (current_time - last_signal_change)))

    # Change the signal if time is up
    if current_time - last_signal_change >= green_duration:
        last_signal_change = current_time
        current_direction_index = (current_direction_index + 1) % len(directions)

        # Reset all signals to Red
        for direction in signal_states:
            signal_states[direction] = 'Red'

        # Set the next direction to Yellow first, then Green
        signal_states[directions[current_direction_index]] = 'Yellow'
        remaining_time = 3  # Yellow light duration

    # If the yellow state duration is over, switch to green
    if signal_states[directions[current_direction_index]] == 'Yellow' and current_time - last_signal_change >= 3:
        signal_states[directions[current_direction_index]] = 'Green'

    # Send the signal state to Arduino
    send_signal_state_to_arduino(directions[current_direction_index], signal_states[directions[current_direction_index]])

    return remaining_time


def combine_frames(frame_north, frame_east, frame_south, frame_west, density):
    """Combine the frames from all directions for display."""
    # Resize each frame to the same size
    small_frame_size = (320, 240)
    frame_north = cv2.resize(frame_north, small_frame_size)
    frame_east = cv2.resize(frame_east, small_frame_size)
    frame_south = cv2.resize(frame_south, small_frame_size)
    frame_west = cv2.resize(frame_west, small_frame_size)

    # Add frame names, signal states, remaining time, and density
    for direction, frame in zip(directions, [frame_north, frame_east, frame_south, frame_west]):
        state = signal_states[direction]
        time_text = f" - {remaining_time}s" if state == 'Green' else ""
        density_text = f"Density: {density[direction]}"
        cv2.putText(frame, f"{direction} ({state}){time_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(frame, density_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Concatenate frames into a single frame
    top_row = np.hstack((frame_north, frame_east))
    bottom_row = np.hstack((frame_south, frame_west))
    combined_frame = np.vstack((top_row, bottom_row))

    return combined_frame


def main():
    global ambulance_detected, ambulance_direction

    cap_south = cv2.VideoCapture('sdf.mp4')
    cap_east = cv2.VideoCapture('t1.mp4')
    cap_north = cv2.VideoCapture('c.mp4')
    cap_west = cv2.VideoCapture('t1.mp4')

    while cap_north.isOpened() and cap_east.isOpened() and cap_south.isOpened() and cap_west.isOpened():
        ret_north, frame_north = cap_north.read()
        ret_east, frame_east = cap_east.read()
        ret_south, frame_south = cap_south.read()
        ret_west, frame_west = cap_west.read()

        if not (ret_north and ret_east and ret_south and ret_west):
            break

        # Detect objects in each direction and count vehicle density
        density = {}
        detected_objects = []
        for direction, frame in zip(directions, [frame_north, frame_east, frame_south, frame_west]):
            objects = detect_objects(frame)
            detected_objects.append(objects)
            vehicle_count = count_objects(objects, vehicle_classes)
            density[direction] = vehicle_count
        # Update traffic signals and get remaining time for the current signal
        remaining_time = update_signal_lamps(density, detected_objects)

        # Combine frames from all directions and display the result
        combined_frame = combine_frames(frame_north, frame_east, frame_south, frame_west, density)
        cv2.imshow('Traffic Signal Control', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_north.release()
    cap_east.release()
    cap_south.release()
    cap_west.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

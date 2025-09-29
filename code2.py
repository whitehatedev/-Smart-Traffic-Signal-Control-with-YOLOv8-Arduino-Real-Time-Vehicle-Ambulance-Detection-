import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load YOLOv8 model
model = YOLO('a.pt')

# Class and vehicle definitions
class_names = [
    'ambulance', 'Barricade', 'Bike', 'Board', 'Building', 'Bullock-cart', 'Bus',
    'Car', 'Cattle', 'Crane', 'Cycle', 'Dog', 'Electricity-Pole', 'Goat',
    'Lamp-Post', 'Manhole', 'Person', 'Rikshaw', 'Road-Divider', 'Tempo',
    'Tractor', 'Traffic-Sign-Board', 'Traffic-Signal', 'Tree', 'Truck', 'Zebra-Crossing'
]

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

# Timing settings
base_signal_duration = 10
min_signal_duration = 5
max_signal_duration = 20
density_scale_factor = 1

# Internal state
last_signal_change = time.time()
current_direction_index = 0
directions = ['North', 'East', 'South', 'West']
remaining_time = base_signal_duration

ambulance_detected = False
ambulance_direction = None


def detect_objects(frame):
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, confidence, cls in zip(detections, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        confidence_percentage = int(confidence * 100)
        class_name = class_names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence_percentage}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return [class_names[int(cls)] for cls in classes]


def count_objects(objects, target_classes):
    return sum(obj in target_classes for obj in objects)


def calculate_signal_time(density):
    if density == 0:
        return 0
    adjusted_time = base_signal_duration + (density * density_scale_factor)
    if density < 5:
        adjusted_time -= 2
    elif density > 10:
        adjusted_time += 3
    return max(min_signal_duration, min(max_signal_duration, int(adjusted_time)))


def update_signal_lamps(density, detected_objects):
    global signal_states, last_signal_change, current_direction_index, remaining_time, ambulance_detected, ambulance_direction

    current_time = time.time()

    for direction, objects in zip(directions, detected_objects):
        if 'ambulance' in objects:
            ambulance_detected = True
            ambulance_direction = direction
            break
    else:
        ambulance_detected = False

    if ambulance_detected:
        for direction in signal_states:
            signal_states[direction] = 'Red'
        signal_states[ambulance_direction] = 'Green'
        remaining_time = max(0, int(base_signal_duration - (current_time - last_signal_change)))
        if remaining_time <= 0:
            ambulance_detected = False
            last_signal_change = current_time
            current_direction_index = (current_direction_index + 1) % len(directions)
        return remaining_time

    current_direction = directions[current_direction_index]
    num_vehicles = density[current_direction]

    if num_vehicles == 0:
        current_direction_index = (current_direction_index + 1) % len(directions)
        remaining_time = 0
        return remaining_time

    green_duration = calculate_signal_time(num_vehicles)
    remaining_time = max(0, int(green_duration - (current_time - last_signal_change)))

    if current_time - last_signal_change >= green_duration:
        last_signal_change = current_time
        current_direction_index = (current_direction_index + 1) % len(directions)
        for direction in signal_states:
            signal_states[direction] = 'Red'
        signal_states[directions[current_direction_index]] = 'Yellow'
        remaining_time = 3

    if signal_states[directions[current_direction_index]] == 'Yellow' and current_time - last_signal_change >= 3:
        signal_states[directions[current_direction_index]] = 'Green'

    return remaining_time


def combine_frames(frame_north, frame_east, frame_south, frame_west, density):
    small_frame_size = (320, 240)
    frame_north = cv2.resize(frame_north, small_frame_size)
    frame_east = cv2.resize(frame_east, small_frame_size)
    frame_south = cv2.resize(frame_south, small_frame_size)
    frame_west = cv2.resize(frame_west, small_frame_size)

    for direction, frame in zip(directions, [frame_north, frame_east, frame_south, frame_west]):
        state = signal_states[direction]
        time_text = f" - {remaining_time}s" if state == 'Green' else ""
        density_text = f"Density: {density[direction]}"
        cv2.putText(frame, f"{direction} ({state}){time_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(frame, density_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    top_row = np.hstack((frame_north, frame_east))
    bottom_row = np.hstack((frame_south, frame_west))
    return np.vstack((top_row, bottom_row))


def main():
    global ambulance_detected, ambulance_direction

    cap_south = cv2.VideoCapture('v.mp4')
    cap_east = cv2.VideoCapture('v.mp4')
    cap_north = cv2.VideoCapture('t1.mp4')
    cap_west = cv2.VideoCapture('c.mp4')

    while cap_north.isOpened() and cap_east.isOpened() and cap_south.isOpened() and cap_west.isOpened():
        ret_north, frame_north = cap_north.read()
        ret_east, frame_east = cap_east.read()
        ret_south, frame_south = cap_south.read()
        ret_west, frame_west = cap_west.read()

        if not (ret_north and ret_east and ret_south and ret_west):
            break

        density = {}
        detected_objects = []
        for direction, frame in zip(directions, [frame_north, frame_east, frame_south, frame_west]):
            objects = detect_objects(frame)
            detected_objects.append(objects)
            vehicle_count = count_objects(objects, vehicle_classes)
            density[direction] = vehicle_count

        remaining_time = update_signal_lamps(density, detected_objects)

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

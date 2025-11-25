import cv2
from ultralytics import YOLO
import os
import datetime
import sys # Import sys for exiting

# --- MODEL SETUP ---
model_path = '../models/best.pt'
violation_output_dir = '../output/violation_snapshots/'
os.makedirs(violation_output_dir, exist_ok=True)

try:
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
    print(f"Model class names: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit() # Exit if model fails to load

# --- CAMERA SELECTION ---
def find_available_cameras(max_check=10):
    """Checks indices 0 to max_check-1 for available cameras."""
    available_cameras = []
    print("Detecting available cameras...")
    for i in range(max_check):
        cap_check = cv2.VideoCapture(i)
        if cap_check.isOpened():
            print(f"  Camera index {i} found.")
            available_cameras.append(i)
            cap_check.release() # Release the camera immediately after checking
        else:
            # If an index doesn't exist, subsequent ones likely won't either
            # You can uncomment the line below to stop checking early
            # break
            pass
    return available_cameras

available_indices = find_available_cameras()

if not available_indices:
    print("Error: No cameras detected. Please ensure a camera is connected and drivers are installed.")
    sys.exit()

chosen_index = -1
if len(available_indices) == 1:
    chosen_index = available_indices[0]
    print(f"Only one camera found (index {chosen_index}). Using it automatically.")
else:
    while True:
        print("\nAvailable camera indices:", available_indices)
        try:
            choice = input("Please enter the index of the camera you want to use: ")
            chosen_index = int(choice)
            if chosen_index in available_indices:
                break # Valid choice
            else:
                print(f"Error: Invalid index. Please choose from {available_indices}.")
        except ValueError:
            print("Error: Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit()


# --- CAMERA SETUP (using chosen index) ---
print(f"Attempting to open camera index {chosen_index}...")
cap = cv2.VideoCapture(chosen_index)
if not cap.isOpened():
    print(f"Error opening video capture device (camera index {chosen_index}).")
    # Add tips for troubleshooting if needed
    print("Troubleshooting tips:")
    print(" - Is the camera being used by another application?")
    print(" - Are the camera drivers installed correctly?")
    print(" - Try restarting the application or your computer.")
    sys.exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

# --- (is_associated helper function remains the same) ---
def is_associated(box_rider, box_other, proximity_threshold=0.1):
    # (Same code as before)
    x1_r, y1_r, x2_r, y2_r = box_rider
    x1_o, y1_o, x2_o, y2_o = box_other
    x_left = max(x1_r, x1_o)
    y_top = max(y1_r, y1_o)
    x_right = min(x2_r, x2_o)
    y_bottom = min(y2_r, y2_o)
    if x_right < x_left or y_bottom < y_top: return False
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    other_area = (x2_o - x1_o) * (y2_o - y1_o)
    if other_area > 0 and (intersection_area / other_area) > proximity_threshold: return True
    center_x_o = (x1_o + x2_o) / 2
    center_y_o = (y1_o + y2_o) / 2
    if (center_x_o > x1_r and center_x_o < x2_r and center_y_o > y1_r and center_y_o < y2_r): return True
    return False

# --- MAIN PROCESSING LOOP ---
print("Starting live camera feed processing... Press 'q' to quit.")
frame_count = 0
while True: # Loop indefinitely until 'q' is pressed
    success, frame = cap.read()
    if not success:
        print("Error reading frame from camera.")
        break

    frame_count += 1

    # --- Run YOLOv8 inference ---
    results = model(frame, verbose=False)

    detected_objects = {
        'rider': [], 'faceWithNoHelmet': [], 'faceWithGoodHelmet': [],
        'faceWithBadHelmet': [], 'numberPlate': []
    }
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name in detected_objects:
            if conf > 0.3:
                detected_objects[class_name].append(((x1, y1, x2, y2), conf))

    # --- VIOLATION RULES ENGINE ---
    riders_processed_this_frame = set()

    for rider_box, rider_conf in detected_objects['rider']:
        rider_tuple = tuple(rider_box)
        if rider_tuple in riders_processed_this_frame:
            continue

        helmet_status = "Unknown"
        helmet_color = (128, 128, 128)
        label = f"Rider ({rider_conf:.2f})"
        is_violation = False

        possible_associations = []
        for h_type in ['faceWithGoodHelmet', 'faceWithBadHelmet', 'faceWithNoHelmet']:
            for h_box, h_conf in detected_objects[h_type]:
                if is_associated(rider_box, h_box):
                    possible_associations.append((h_type, h_conf, h_box))

        if possible_associations:
            possible_associations.sort(key=lambda x: x[1], reverse=True)
            detected_helmet_type = possible_associations[0][0]
            best_helmet_conf = possible_associations[0][1]

            if detected_helmet_type == 'faceWithGoodHelmet':
                helmet_status = "Good Helmet"
                helmet_color = (0, 255, 0)
                label = f"Rider: OK ({best_helmet_conf:.2f})"
            elif detected_helmet_type == 'faceWithBadHelmet':
                helmet_status = "Bad Helmet (Violation)"
                helmet_color = (0, 255, 255)
                label = f"Rider: BAD HELMET ({best_helmet_conf:.2f})"
                is_violation = True
            elif detected_helmet_type == 'faceWithNoHelmet':
                helmet_status = "No Helmet (Violation)"
                helmet_color = (0, 0, 255)
                label = f"Rider: NO HELMET ({best_helmet_conf:.2f})"
                is_violation = True
        else:
             label = f"Rider: Helmet ?"

        # Draw the box
        cv2.rectangle(frame, (rider_box[0], rider_box[1]), (rider_box[2], rider_box[3]), helmet_color, 2)
        cv2.putText(frame, label, (rider_box[0], rider_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, helmet_color, 2)

        # --- SAVE VIOLATION SNAPSHOTS ---
        if is_violation:
            riders_processed_this_frame.add(rider_tuple)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            violation_type = helmet_status.split(" ")[0]

            snapshot_filename = f"{timestamp}_{violation_type}Helmet_live.jpg"
            snapshot_path = os.path.join(violation_output_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)

            best_plate_conf = -1
            associated_plate_box = None
            for plate_box, plate_conf in detected_objects['numberPlate']:
                if is_associated(rider_box, plate_box, proximity_threshold=0.05):
                    if plate_conf > best_plate_conf:
                        best_plate_conf = plate_conf
                        associated_plate_box = plate_box

            if associated_plate_box:
                p_x1, p_y1, p_x2, p_y2 = associated_plate_box
                padding = 5
                p_x1 = max(0, p_x1 - padding)
                p_y1 = max(0, p_y1 - padding)
                p_x2 = min(frame_width, p_x2 + padding)
                p_y2 = min(frame_height, p_y2 + padding)
                plate_crop = frame[p_y1:p_y2, p_x1:p_x2]

                if plate_crop.size > 0:
                    plate_filename = f"{timestamp}_{violation_type}Helmet_plate_live.jpg"
                    plate_path = os.path.join(violation_output_dir, plate_filename)
                    cv2.imwrite(plate_path, plate_crop)

    # --- Display the live feed ---
    cv2.imshow("Live Helmet Violation Detection (Press 'q' to quit)", frame) # Added instruction to title

    # --- Check for 'q' key press to exit ---
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

# --- CLEANUP ---
print("Releasing camera and closing windows...")
cap.release()
cv2.destroyAllWindows()
print(f"Violation snapshots saved in: {violation_output_dir}")
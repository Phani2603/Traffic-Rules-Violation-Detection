import cv2
from ultralytics import YOLO
import os

# --- MODEL AND VIDEO SETUP ---

# !!! IMPORTANT: UPDATE THIS PATH !!!
# Path to your trained model file
model_path = '../models/best.pt' # Assuming you moved best.pt to the models folder

# Define the path to your input video file
video_path = '../input_data/traffic1.mp4'  # Assuming video is in input_data folder

# Define the output path for the annotated video
output_dir = '../output/annotated_videos/'
os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
output_video_path = os.path.join(output_dir, 'traffic_annotated1.mp4')

# Load your custom-trained, 5-class model
try:
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
    print(f"Model class names: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model path is correct and the .pt file is valid.")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

# Get video properties for saving
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
print(f"Output video will be saved to: {output_video_path}")

# --- HELPER FUNCTION (To check if two boxes are close/overlapping) ---
def is_associated(box_rider, box_other, proximity_threshold=0.1):
    """
    Checks if box_other (e.g., a face/helmet) is spatially associated with box_rider.
    A simple check: Is the center of the 'other' box inside the 'rider' box?
    Improved: Check overlap based on Intersection over Area of the 'other' box.
    """
    x1_r, y1_r, x2_r, y2_r = box_rider
    x1_o, y1_o, x2_o, y2_o = box_other

    # Calculate intersection coordinates
    x_left = max(x1_r, x1_o)
    y_top = max(y1_r, y1_o)
    x_right = min(x2_r, x2_o)
    y_bottom = min(y2_r, y2_o)

    # If no overlap
    if x_right < x_left or y_bottom < y_top:
        return False

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of the 'other' box
    other_area = (x2_o - x1_o) * (y2_o - y1_o)

    # Check if intersection area is a significant portion of the 'other' box area
    if other_area > 0 and (intersection_area / other_area) > proximity_threshold:
        return True

    # Fallback: check center point inclusion (useful for small contained boxes)
    center_x_o = (x1_o + x2_o) / 2
    center_y_o = (y1_o + y2_o) / 2
    if (center_x_o > x1_r and center_x_o < x2_r and
            center_y_o > y1_r and center_y_o < y2_r):
        return True

    return False

# --- MAIN PROCESSING LOOP ---
print("Starting video processing...")
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video reached or error reading frame.")
        break

    frame_count += 1
    if frame_count % 10 == 0: # Print progress every 10 frames
        print(f"Processing frame {frame_count}...")

    # Run YOLOv8 inference on the frame
    # 'verbose=False' prevents excessive printing from Ultralytics
    results = model(frame, verbose=False)

    # Dictionary to store detected objects for this frame
    detected_objects = {
        'rider': [],
        'faceWithNoHelmet': [],
        'faceWithGoodHelmet': [],
        'faceWithBadHelmet': [],
        'numberPlate': []
    }

    # Extract all boxes and sort them by class
    # Use results[0].boxes to access detection details
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in detected_objects:
            detected_objects[class_name].append(((x1, y1, x2, y2), conf)) # Store box and confidence

    # --- VIOLATION RULES ENGINE ---

    # Iterate over every detected rider
    for rider_box, rider_conf in detected_objects['rider']:
        helmet_status = "Unknown"
        helmet_color = (128, 128, 128) # Default GRAY
        label = f"Rider ({rider_conf:.2f})" # Include confidence in label

        best_helmet_conf = -1
        detected_helmet_type = None

        # Check for associated helmet/face detections
        possible_associations = []
        for h_type in ['faceWithGoodHelmet', 'faceWithBadHelmet', 'faceWithNoHelmet']:
            for h_box, h_conf in detected_objects[h_type]:
                if is_associated(rider_box, h_box):
                    possible_associations.append((h_type, h_conf, h_box))

        # Determine the most confident association for this rider
        if possible_associations:
            # Sort by confidence (highest first)
            possible_associations.sort(key=lambda x: x[1], reverse=True)
            detected_helmet_type = possible_associations[0][0] # Get type of most confident match
            best_helmet_conf = possible_associations[0][1]

            if detected_helmet_type == 'faceWithGoodHelmet':
                helmet_status = "Good Helmet"
                helmet_color = (0, 255, 0) # GREEN
                label = f"Rider: OK ({best_helmet_conf:.2f})"
            elif detected_helmet_type == 'faceWithBadHelmet':
                helmet_status = "Bad Helmet (Violation)"
                helmet_color = (0, 255, 255) # YELLOW
                label = f"Rider: BAD HELMET ({best_helmet_conf:.2f})"
                # --- Add violation logging here if needed ---
            elif detected_helmet_type == 'faceWithNoHelmet':
                helmet_status = "No Helmet (Violation)"
                helmet_color = (0, 0, 255) # RED
                label = f"Rider: NO HELMET ({best_helmet_conf:.2f})"
                # --- Add violation logging here if needed ---
        else:
            # No associated helmet/face found for this rider
             label = f"Rider: Helmet ?" # Keep rider label simple if unknown


        # Draw the box around the rider with status color
        cv2.rectangle(frame, (rider_box[0], rider_box[1]), (rider_box[2], rider_box[3]), helmet_color, 2)
        cv2.putText(frame, label, (rider_box[0], rider_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, helmet_color, 2)

        # Optional: Draw boxes for associated helmets/faces/plates
        # (Can make the output crowded, enable if needed for debugging)
        # for h_type, h_conf, h_box in possible_associations:
        #     cv2.rectangle(frame, (h_box[0], h_box[1]), (h_box[2], h_box[3]), (255, 0, 0), 1) # Blue box for associated item
        # for plate_box, plate_conf in detected_objects['numberPlate']:
        #     if is_associated(rider_box, plate_box, proximity_threshold=0.05): # Lower threshold for plates
        #          cv2.rectangle(frame, (plate_box[0], plate_box[1]), (plate_box[2], plate_box[3]), (255, 255, 0), 1) # Cyan for plate


    # Write the annotated frame to the output video file
    out.write(frame)

    # Display the annotated frame (optional, comment out for faster processing without display)
    # cv2.imshow("Helmet Violation Detection", frame)

    # Break the loop if 'q' is pressed (only works if cv2.imshow is active)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #    break

# --- CLEANUP ---
print("Finished processing. Releasing resources...")
cap.release()
out.release()
# cv2.destroyAllWindows() # Only needed if cv2.imshow was used
print(f"Annotated video saved to: {output_video_path}")
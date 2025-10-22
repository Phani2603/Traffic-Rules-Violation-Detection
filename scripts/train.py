from ultralytics import YOLO
import os

# --- 1. DEFINE YOUR DATA ---

# These are the correct, 0-indexed class names based on your list
class_names = [
    'numberPlate',
    'faceWithNoHelmet',
    'faceWithGoodHelmet',
    'faceWithBadHelmet',
    'rider'
]

# The directory paths *relative to this script*
# This assumes your 'dataset' folder is one level above your 'scripts' folder
data_paths = {
    'train': '../dataset/train/images',
    'val': '../dataset/valid/images'
}

# --- 2. CREATE THE data.yaml FILE ---

# This file tells YOLOv8 where your data is and what it's called
yaml_content = f"""
train: {data_paths['train']}
val: {data_paths['val']}

nc: {len(class_names)}
names: {class_names}
"""

# Write the content to a file named 'data.yaml' *in this same scripts folder*
with open('data.yaml', 'w') as f:
    f.write(yaml_content)

print("--- 'data.yaml' created successfully (without test set) ---")
print(yaml_content)
print("----------------------------------------------------------")


# --- 3. LOAD MODEL AND START TRAINING ---

# Load a pre-trained model. 'yolov8n.pt' is the smallest and fastest.
model = YOLO('yolov8n.pt')

# Start the training!
print("Starting model training... This may take a while. 🚀")
results = model.train(
    data='data.yaml',       # Path to our new YAML file
    epochs=50,              # Start with 50 epochs. Increase to 100 for better results.
    imgsz=640,              # Resize all images to 640x640
    project='training_results',  # Save results in a new 'training_results' folder
    name='helmet_run1',       # Name this specific training run
    exist_ok=True
)

print("--- Training complete! ---")
print(f"Model and results saved to: {results.save_dir}")
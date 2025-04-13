import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import random  # for random selection
import torch

def get_rotation_matrix(angle, img_w, img_h):
    """
    Compute the rotation matrix for a given angle (in degrees) around the center of the image.
    """
    center = (img_w / 2, img_h / 2)
    return cv2.getRotationMatrix2D(center, angle, 1)

def rotate_bounding_box_with_M(x_center, y_center, width, height, M, img_w, img_h):
    """
    Rotate a YOLO-format bounding box using a given transformation matrix M.
    The bounding box is specified with normalized coordinates (x_center, y_center, width, height).
    Returns the new bounding box in normalized format.
    """
    # Convert YOLO normalized coordinates to absolute pixel coordinates
    abs_x_center = x_center * img_w
    abs_y_center = y_center * img_h
    abs_width = width * img_w
    abs_height = height * img_h

    # Compute the four corners of the original bounding box
    x1 = abs_x_center - abs_width / 2
    y1 = abs_y_center - abs_height / 2
    x2 = abs_x_center + abs_width / 2
    y2 = y1
    x3 = x2
    y3 = abs_y_center + abs_height / 2
    x4 = x1
    y4 = y3

    pts = np.array([[x1, y1],
                    [x2, y2],
                    [x3, y3],
                    [x4, y4]], dtype=np.float32).reshape(-1, 1, 2)

    # Apply the rotation matrix M to the bounding box corners
    pts_rotated = cv2.transform(pts, M).reshape(-1, 2)

    # Compute the new axis-aligned bounding box from the rotated corners
    x_min, y_min = np.min(pts_rotated, axis=0)
    x_max, y_max = np.max(pts_rotated, axis=0)

    new_abs_x_center = (x_min + x_max) / 2.0
    new_abs_y_center = (y_min + y_max) / 2.0
    new_abs_width = x_max - x_min
    new_abs_height = y_max - y_min

    # Convert back to normalized YOLO format
    new_x_center = new_abs_x_center / img_w
    new_y_center = new_abs_y_center / img_h
    new_width = new_abs_width / img_w
    new_height = new_abs_height / img_h

    # Clip values to [0, 1] to avoid negative or >1 coordinates
    new_x_center = np.clip(new_x_center, 0, 1)
    new_y_center = np.clip(new_y_center, 0, 1)
    new_width = np.clip(new_width, 0, 1)
    new_height = np.clip(new_height, 0, 1)

    return new_x_center, new_y_center, new_width, new_height

def process_label_file(input_label_path, output_label_path, M, img_w, img_h):
    """
    Process a label file by rotating each bounding box using transformation matrix M.
    The label file is assumed to be in YOLO format:
        <class> <x_center> <y_center> <width> <height>
    with normalized coordinates.
    """
    new_lines = []
    with open(input_label_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 5:
                continue
            cls = tokens[0]
            x_center, y_center, width, height = map(float, tokens[1:])
            new_x, new_y, new_w, new_h = rotate_bounding_box_with_M(x_center, y_center, width, height, M, img_w, img_h)
            new_line = f"{cls} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n"
            new_lines.append(new_line)
    with open(output_label_path, "w") as f:
        f.writelines(new_lines)

def draw_bounding_boxes(image, label_path, color=(0, 255, 0)):
    """Draw bounding boxes from YOLO format labels on the image."""
    img_h, img_w = image.shape[:2]
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return image
    with open(label_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 5:
                continue
            cls, x_center, y_center, width, height = tokens
            x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
            x_center_px = int(x_center * img_w)
            y_center_px = int(y_center * img_h)
            width_px = int(width * img_w)
            height_px = int(height * img_h)
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, "Armor", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def augment_image_with_rotation(image, M):
    """
    Rotate the image using the given transformation matrix M.
    """
    rows, cols = image.shape[:2]
    return cv2.warpAffine(image, M, (cols, rows))

def cleanup_rotated_files(directory, prefix="rotated_"):
    """
    Remove any files in the given directory that start with the specified prefix.
    """
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Removed {file_path}")
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")

# =========================
# Data Augmentation Section
# =========================

# Define original images and labels directories.
img_dir = "datasets/robot_detection/train/images/"
lbl_dir = "datasets/robot_detection/train/labels/"

# Clean up previously augmented files to avoid duplicate rotations.
cleanup_rotated_files(img_dir, prefix="rotated_")
cleanup_rotated_files(lbl_dir, prefix="rotated_")

# Get list of all images in the original images folder.
all_images = os.listdir(img_dir)

already_displayed = False

for image_name in all_images:
    image_path = os.path.join(img_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_name}")
        continue

    img_h, img_w = image.shape[:2]
    # Generate a random rotation angle between -8 and 8 degrees.
    rotation_angle = np.random.choice([1, -1]) * np.random.rand() * 8
    M = get_rotation_matrix(rotation_angle, img_w, img_h)
    rotated = augment_image_with_rotation(image, M)
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    
    # Add noise
    noise = np.random.normal(0, 0.3, image.shape).astype(np.uint8)
    noisy = cv2.addWeighted(image, 1, noise, 0.5, 0)

    # Save the augmented image in the same folder.
    base, ext = os.path.splitext(image_name)
    new_image_name = f"rotated_{base}{ext}"
    new_image_path = os.path.join(img_dir, new_image_name)
    cv2.imwrite(new_image_path, rotated)

    new_image_name = f"bright_{base}{ext}"
    new_image_path = os.path.join(img_dir, new_image_name)
    cv2.imwrite(new_image_path, bright)
    with open(image_path, "r") as file_in:
        with open(new_image_path, "w") as file_out:
            for line in file_in:
                file_out.write(line)

    new_image_name = f"noisy_{base}{ext}"
    new_image_path = os.path.join(img_dir, new_image_name)
    cv2.imwrite(new_image_path, noisy)
    with open(image_path, "r") as file_in:
        with open(new_image_path, "w") as file_out:
            for line in file_in:
                file_out.write(line)

    # Process labels: save new labels in the same labels folder.
    input_label_path = os.path.join(lbl_dir, base + ".txt")
    new_label_path = os.path.join(lbl_dir, f"rotated_{base}.txt")
    if os.path.exists(input_label_path):
        process_label_file(input_label_path, new_label_path, M, img_w, img_h)
    else:
        print(f"No label file found for {image_name}")

    # Display the first example only.
    if not already_displayed:
        image_with_boxes = draw_bounding_boxes(image.copy(), input_label_path)
        rotated_with_boxes = draw_bounding_boxes(rotated.copy(), new_label_path)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title("Original + BBoxes")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(rotated_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Rotated ({rotation_angle:.2f}°) + BBoxes")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        already_displayed = True

print("Data augmentation completed.")

# ======================================
# YOLOv5 Training and Validation Section
# ======================================

# Create a minimal data config file for YOLOv5 training.
# Since YOLOv5 runs from its own directory, use relative paths pointing back to your dataset.
data_yaml_path = "datasets/robot_detection/data.yaml"
if not os.path.exists(data_yaml_path):
    data_yaml_content = """
train: ../datasets/robot_detection/train/images
val: ../datasets/robot_detection/train/images
nc: 1
names: ['Armor']
    """.strip()
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)
    print(f"Created {data_yaml_path} for YOLOv5 training.")

# Train the YOLOv5 model.
train_command = [
    "python", "yolov5/train.py",
    "--img", "640",
    "--batch", "16",
    "--epochs", "50",
    "--data", data_yaml_path,
    "--weights", "yolov5s.pt",  # Pre-trained YOLOv5 small model.
    "--cache", "ram",
    "--project", "runs/train",
    "--name", "robot_detection_experiment"
]
print("Starting YOLOv5 training...")
subprocess.run(train_command)

# Check if training produced the weights file before starting validation.
weights_path = os.path.join("runs/train/robot_detection_experiment/weights", "best.pt")
if not os.path.exists(weights_path):
    print("Training did not produce best.pt. Skipping validation.")
else:
    test_command = [
        "python", "yolov5/val.py",
        "--weights", weights_path,
        "--data", data_yaml_path,
        "--img", "640",
        "--conf", "0.001"
    ]
    print("Starting YOLOv5 validation...")
    subprocess.run(test_command)

# ======================================
# Model Testing on a Random Train Image
# ======================================

# Pick one random image from the training images directory.
train_images = os.listdir(img_dir)
test_img_name = random.choice(train_images)
test_img_path = os.path.join(img_dir, test_img_name)

# Load the test image.
test_image = cv2.imread(test_img_path)
if test_image is None:
    print(f"Failed to load test image: {test_img_name}")
else:
    # Load the trained YOLOv5 model using the best.pt weights.
    # This uses the torch.hub interface to load a custom model.
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
    model.conf = 0.25  # Set a confidence threshold if desired.

    # Run inference on the test image.
    results = model(test_img_path)
    pred_df = results.pandas().xyxy[0]  # Get predictions as a pandas DataFrame.

    # Define a helper function to draw prediction bounding boxes.
    def draw_predictions(image, df, color=(0, 0, 255)):
        for index, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            cls = row['name'] if 'name' in row else 'pred'
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    # Draw the predicted bounding boxes (in red) on a copy of the test image.
    pred_image = test_image.copy()
    pred_image = draw_predictions(pred_image, pred_df, color=(0, 0, 255))

    # Draw the ground truth bounding boxes (in green) using your draw_bounding_boxes function.
    base_name, ext = os.path.splitext(test_img_name)
    true_label_path = os.path.join(lbl_dir, base_name + ".txt")
    true_image = test_image.copy()
    true_image = draw_bounding_boxes(true_image, true_label_path, color=(0, 255, 0))

    # Display the model's predictions and the ground truth side by side.
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB))
    plt.title("Model Predictions")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(true_image, cv2.COLOR_BGR2RGB))
    plt.title("Ground Truth Labels")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


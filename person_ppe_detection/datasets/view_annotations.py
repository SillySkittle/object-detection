import os
import cv2
import random


images_path = 'datasets/augmented_data/cropped/images'
labels_path = 'datasets/augmented_data/cropped/labels'
output_path = 'datasets/augmented_data/cropped/annotated_images'
classes_file = 'datasets/cropped_classes.txt'

# Load class names
with open(classes_file, 'r') as f:
    class_names = f.read().strip().split('\n')

# Define specific colors for the classes
colors = [
    (0, 0, 255),  # Red
    (0, 255, 255),  # Yellow
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (255, 0, 255),  # Pink
    (64, 224, 208),  # Turquoise
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (255, 255, 255)  # White
]

# Assign colors to classes (loop around if more classes than colors)
class_colors = {class_name: colors[i % len(colors)] for i, class_name in enumerate(class_names)}

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)


# Function to draw bounding boxes on the image
def draw_bounding_boxes(img, label_file):
    height, width, _ = img.shape
    with open(label_file, 'r') as f:
        labels = f.readlines()

    for label in labels:
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, label.strip().split())
        class_id = int(class_id)

        # Convert YOLO format to bounding box coordinates
        x1 = int((x_center - bbox_width / 2) * width)
        y1 = int((y_center - bbox_height / 2) * height)
        x2 = int((x_center + bbox_width / 2) * width)
        y2 = int((y_center + bbox_height / 2) * height)

        # Draw the bounding box
        class_name = class_names[class_id]
        color = class_colors[class_name]
        thickness = 2
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        img = cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)

    return img


# Process images
for image_file in os.listdir(images_path):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(images_path, image_file)
        label_file = os.path.join(labels_path, os.path.splitext(image_file)[0] + '.txt')

        if os.path.exists(label_file):
            img = cv2.imread(image_path)
            img_with_boxes = draw_bounding_boxes(img, label_file)
            output_image_path = os.path.join(output_path, image_file)
            cv2.imwrite(output_image_path, img_with_boxes)

print(f"Annotated images have been saved to {output_path}")
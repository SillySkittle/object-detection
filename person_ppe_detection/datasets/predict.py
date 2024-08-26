import cv2
import numpy as np
from ultralytics import YOLO
import os

# Paths
weights_path = 'person_detection.pt'  # Path to your YOLOv8 weights file
input_dir = 'test2'  # Directory containing test images
output_dir = 'prediction2'  # Directory to save annotated images

# Load the model
model = YOLO(weights_path)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# Function to draw bounding boxes on image
def draw_boxes(image, detections):
    for det in detections:
        # Get bounding box coordinates, confidence score, and class ID
        x1, y1, x2, y2, conf, cls = det[:6]
        color = (0, 255, 0)  # Green color for bounding boxes
        label = f'{model.names[int(cls)]} {conf:.2f}'

        # Draw rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        # Put text
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


# Process each image in the input directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    output_path = os.path.join(output_dir, image_name)

    # Load image
    image = cv2.imread(image_path)

    # Predict
    results = model(image)

    # results is a list of Result objects
    for result in results:
        # Get the detections
        detections = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2) in normalized coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Combine detections, scores, and class IDs into a single list
        detections_combined = np.hstack((detections, scores[:, np.newaxis], class_ids[:, np.newaxis]))

        # Draw boxes on image
        annotated_image = draw_boxes(image, detections_combined)

        # Save the annotated image
        cv2.imwrite(output_path, annotated_image)

    print(f'Processed {image_name}')

print('Processing complete.')
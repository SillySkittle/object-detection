import cv2
from ultralytics import YOLO
import os
from pathlib import Path

# Paths to model weights
person_model_path = 'person_detection.pt'
ppe_model_path = 'ppe_detection.pt'
output_folder = 'inference'
input_image_path = 'test/005268.jpg'


# Load models
person_model = YOLO(person_model_path)
ppe_model = YOLO(ppe_model_path)

# Ensure output folder exists
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Load and preprocess the image
image = cv2.imread(input_image_path)
original_image = image.copy()

# Perform inference using the person detection model
results = person_model(image)
results = results[0]  # The output is a list of results, get the first result

# Extract bounding boxes
person_bboxes = results.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
person_scores = results.boxes.conf.cpu().numpy()  # Confidence scores
person_classes = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs

# For each detected person
for i, bbox in enumerate(person_bboxes):
    x1, y1, x2, y2 = bbox
    conf = person_scores[i]
    cls = person_classes[i]
    person_image = original_image[int(y1):int(y2), int(x1):int(x2)]

    # Save the cropped person image
    person_image_path = os.path.join(output_folder, f'person_{i}.jpg')
    cv2.imwrite(person_image_path, person_image)

    # Perform PPE detection on the cropped person image
    ppe_results = ppe_model(person_image)
    ppe_results = ppe_results[0]  # The output is a list of results, get the first result

    # Extract bounding boxes for PPE
    ppe_bboxes = ppe_results.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
    ppe_scores = ppe_results.boxes.conf.cpu().numpy()  # Confidence scores
    ppe_classes = ppe_results.boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Draw bounding boxes and confidence scores on the cropped person image
    for ppe_bbox in zip(ppe_bboxes, ppe_scores, ppe_classes):
        bbox, score, class_id = ppe_bbox
        x1, y1, x2, y2 = bbox
        cv2.rectangle(person_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(person_image, f'{ppe_model.names[int(class_id)]} {score:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image with PPE detection results
    output_person_image_path = os.path.join(output_folder, f'person_{i}_with_ppe.jpg')
    cv2.imwrite(output_person_image_path, person_image)

# Draw bounding boxes and confidence scores on the original image
for bbox, conf, cls in zip(person_bboxes, person_scores, person_classes):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(original_image, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Save the original image with person detection results
output_original_image_path = os.path.join(output_folder, 'original_with_persons.jpg')
cv2.imwrite(output_original_image_path, original_image)

print("Inference complete. Results saved in:", output_folder)
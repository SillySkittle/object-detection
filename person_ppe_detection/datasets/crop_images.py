import cv2
import os
import argparse

def adjust_bbox(bbox, person_bbox):
    # Adjust bbox based on person_bbox (person's cropped region)
    x_center, y_center, width, height = bbox
    px, py, pw, ph = person_bbox

    # Calculate the new bounding box coordinates relative to the cropped image
    new_x_center = (x_center - px + pw / 2) / pw
    new_y_center = (y_center - py + ph / 2) / ph
    new_width = width / pw
    new_height = height / ph

    # Ensure that the new bounding box coordinates are within the bounds of the cropped image
    if new_x_center < 0 or new_x_center > 1 or new_y_center < 0 or new_y_center > 1:
        return None

    # Ensure the width and height are within valid range
    if new_width <= 0 or new_width > 1 or new_height <= 0 or new_height > 1:
        return None

    return [new_x_center, new_y_center, new_width, new_height]

def process_image(image_path, annotation_path, output_image_dir, output_label_dir, person_class_id=0):
    # Load image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Load annotations
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            annotations.append([int(class_id), x_center * w, y_center * h, width * w, height * h])

    # Extract person bounding boxes
    person_bboxes = [ann for ann in annotations if ann[0] == person_class_id]

    # Process each person bounding box
    for i, person_bbox in enumerate(person_bboxes):
        px_center, py_center, p_width, p_height = person_bbox[1:]

        # Crop the image around the person bbox
        x1 = int(px_center - p_width / 2)
        y1 = int(py_center - p_height / 2)
        x2 = int(px_center + p_width / 2)
        y2 = int(py_center + p_height / 2)

        # Ensure cropping coordinates are within the image bounds
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)

        # Calculate the width and height of the cropped image
        crop_w = x2 - x1
        crop_h = y2 - y1

        # Crop the image if the bounding box is within the valid area
        if crop_w > 0 and crop_h > 0:
            cropped_image = image[y1:y2, x1:x2]

            # Create new annotations for cropped image
            new_annotations = []
            for ann in annotations:
                if ann[0] != person_class_id:  # Skip person bbox
                    # Adjust the bounding box coordinates to the cropped image
                    adjusted_bbox = adjust_bbox(ann[1:], person_bbox[1:])
                    if adjusted_bbox:
                        new_class_id = ann[0] - 1  # Decrease class index by 1
                        new_annotations.append([new_class_id] + adjusted_bbox)

            if new_annotations:
                # Save the cropped image
                output_image_path = os.path.join(output_image_dir,
                                                 f"{os.path.splitext(os.path.basename(image_path))[0]}_p{i + 1}.jpg")
                cv2.imwrite(output_image_path, cropped_image)

                # Save the corresponding annotation
                output_label_path = os.path.join(output_label_dir,
                                                 f"{os.path.splitext(os.path.basename(annotation_path))[0]}_p{i + 1}.txt")
                with open(output_label_path, 'w') as f:
                    for ann in new_annotations:
                        f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description='Process and crop images with bounding boxes.')
    parser.add_argument('image_dir', type=str, help='Directory containing input images.')
    parser.add_argument('label_dir', type=str, help='Directory containing input annotations.')
    parser.add_argument('output_image_dir', type=str, help='Directory to save cropped images.')
    parser.add_argument('output_label_dir', type=str, help='Directory to save cropped annotations.')
    parser.add_argument('--person_class_id', type=int, default=0, help='Class ID for person annotations (default: 0).')

    args = parser.parse_args()

    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_label_dir, exist_ok=True)

    for image_file in os.listdir(args.image_dir):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(args.image_dir, image_file)
            annotation_path = os.path.join(args.label_dir, f"{os.path.splitext(image_file)[0]}.txt")
            process_image(image_path, annotation_path, args.output_image_dir, args.output_label_dir, args.person_class_id)

if __name__ == "__main__":
    main()

# python datasets/crop_images.py datasets/augmented_data/images datasets/augmented_data/all_labels datasets/augmented_data/cropped/images datasets/augmented_data/cropped/labels
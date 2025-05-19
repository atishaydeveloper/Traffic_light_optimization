# vehicle_detection.py
import cv2
import numpy as np

# Define vehicle classes of interest
VEHICLE_CLASSES = ["car", "motorbike", "bus", "bicycle", "truck"]

def load_yolo_model(weights_path="yolov4.weights", cfg_path="yolov4.cfg"):
    """Loads the YOLOv4 model from disk."""
    try:
        net = cv2.dnn.readNet(weights_path, cfg_path)
        # Optional: Set preferable backend and target if CUDA is available
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return net
    except cv2.error as e:
        print(f"Error loading YOLO model: {e}")
        print(f"Ensure '{weights_path}' and '{cfg_path}' exist.")
        return None

def load_class_names(names_path="coco.names"):
    """Loads class names from the coco.names file."""
    try:
        with open(names_path, "r") as f:
            classes = f.read().strip().split("\n")
        return classes
    except FileNotFoundError:
        print(f"Error: Class names file not found at '{names_path}'")
        return None

def detect_vehicles(image_np, net, all_classes, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Detects vehicles in an image using the YOLOv4 model.

    Args:
        image_np (np.ndarray): The input image as a NumPy array (BGR format).
        net: The loaded YOLOv4 DNN network.
        all_classes (list): A list of all class names from coco.names.
        confidence_threshold (float): Minimum probability to filter weak detections.
        nms_threshold (float): Threshold used in Non-Maximum Suppression.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The image with bounding boxes drawn (BGR format).
            - dict: A dictionary with counts of detected vehicle types.
            - list: A list of detected bounding boxes details [(x, y, w, h, label), ...].
    """
    if net is None or all_classes is None:
        return image_np, {cls: 0 for cls in VEHICLE_CLASSES}, []

    height, width, _ = image_np.shape
    annotated_image = image_np.copy() # Work on a copy to preserve original
    vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}
    detection_details = []

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get output layer names and run forward pass
    try:
        output_layers_names = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(output_layers_names)
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        return annotated_image, vehicle_counts, detection_details

    boxes = []
    confidences = []
    class_ids = []

    # Process detections from all output layers
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = all_classes[class_id]

            # Filter by confidence and if the class is a vehicle we care about
            if confidence > confidence_threshold and label in VEHICLE_CLASSES:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) > 0:
        # Flatten indices if it's nested
        if isinstance(indices, tuple):
             indices = indices[0] # Handle potential tuple output
        if hasattr(indices, 'flatten'):
            indices = indices.flatten()


        for i in indices:
            try:
                 box_index = i # Use the index directly if flatten() returns indices
            except TypeError:
                 box_index = i[0] # Handle cases where indices might be nested arrays

            x, y, w, h = boxes[box_index]
            label = all_classes[class_ids[box_index]]
            confidence = confidences[box_index]

            # Increment count
            if label in vehicle_counts:
                vehicle_counts[label] += 1

            # Store detection details
            detection_details.append((x, y, w, h, label))

            # Draw bounding box and label
            color = (0, 255, 0) # Green for bounding boxes
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(annotated_image, text, (x, y - 10 if y > 20 else y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated_image, vehicle_counts, detection_details
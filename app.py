import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 detection model
model = YOLO('runs/detect/train/weights/best.pt')  # Replace with your actual trained detection model

# Define hair type labels (adjust according to your dataset)
hair_type_labels = ['Straight', 'Wavy', 'Curly', 'Kinky', 'Dreadlocks']

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform YOLOv8 inference on the frame (detect hair types with bounding boxes)
    results = model(frame)

    if results and len(results) > 0:
        print(f"Detections: {len(results)}")
        
        # Loop over the detections and draw bounding boxes with labels
        for r in results:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates (top-left and bottom-right)

                    print(f"Bounding box coordinates: {x1, y1, x2, y2}")
                    
                    # Get class ID and confidence score
                    class_id = int(box.cls[0]) if box.cls is not None else 0
                    confidence = float(box.conf[0]) if box.conf is not None else 0

                    print(f"Class ID: {class_id}, Confidence: {confidence:.2f}")

                    # Get the predicted hair type label
                    hair_type = hair_type_labels[class_id]
                    print(f"Detected: {hair_type} with confidence {confidence:.2f}")

                    # Draw a bounding box around the detected hair
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Prepare the label with confidence score
                    label = f"{hair_type} ({confidence:.2f})"

                    # Draw the label above the bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                print("No bounding boxes found in results.")
    else:
        print("No detections found.")

    # Display the final frame with annotations
    cv2.imshow('Hair Type Detection', frame)

    # Handle window close event (manual window close or 'q' key press)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Hair Type Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

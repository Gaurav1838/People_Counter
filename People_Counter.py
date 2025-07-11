import cv2
import numpy as np

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt',
                               'MobileNetSSD_deploy.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture('sample.mp4')  # Use 0 for webcam

# Video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Line position (vertical center)
line_position = frame_width // 2

# Count values
entry_count = 0
exit_count = 0

# Store previous object centers
trackers = []

def update_trackers(current_centers):
    global entry_count, exit_count, trackers

    updated = []

    for new_cx, new_cy in current_centers:
        matched = False
        for old_cx, old_cy in trackers:
            dist = np.linalg.norm(np.array((new_cx, new_cy)) - np.array((old_cx, old_cy)))
            if dist < 50:
                # Direction check
                if old_cx < line_position and new_cx >= line_position:
                    entry_count += 1
                elif old_cx > line_position and new_cx <= line_position:
                    exit_count += 1
                matched = True
                updated.append((new_cx, new_cy))
                break

        if not matched:
            updated.append((new_cx, new_cy))

    trackers = updated

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw vertical counting line
    cv2.line(frame, (line_position, 0), (line_position, frame_height), (255, 0, 0), 2)

    # Detect people using MobileNet-SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    current_centers = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array(
                [frame_width, frame_height, frame_width, frame_height])
            (startX, startY, endX, endY) = box.astype("int")
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)
            current_centers.append((centerX, centerY))

            # Draw person box and center
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), -1)

    # Update trackers for direction counting
    update_trackers(current_centers)

    # Display counts
    cv2.putText(frame, f'Entry: {entry_count}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Exit: {exit_count}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Bidirectional People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

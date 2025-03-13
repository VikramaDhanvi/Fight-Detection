import torch
import cv2
import numpy as np

# Load YOLOv5 model (small version) using PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open a video capture object from a video file
cap = cv2.VideoCapture(0)

# Previous frame for motion analysis
prev_frame = None

# Flag to keep track of potential fight detection
potential_fight = False

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Break the loop if there are no more frames
    if not ret:
        break

    # Perform object detection using YOLOv5
    results = model(frame)

    # Check for potential fight based on motion analysis
    if prev_frame is not None:
        # Convert frames to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference between frames
        frame_diff = cv2.absdiff(gray_prev_frame, gray_frame)

        # Threshold the difference to identify significant motion
        _, thresh_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Count the number of non-zero pixels (indicating motion)
        motion_count = np.count_nonzero(thresh_diff)

        # If motion count is above a threshold, consider it a potential fight
        if motion_count > 145000:  # Adjust threshold as needed
            potential_fight = True
        else:
            potential_fight = False

    # Counter for the number of people
    num_people = 0

    # Iterate through detected objects
    for index, row in results.pandas().xyxy[0].iterrows():
        label = row['name']

        # Check if the detected object is a person
        if label == 'person':
            # Get confidence score for the 'person' class
            confidence = row['confidence']

            # Extract coordinates of the person
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            # Change color based on potential fight
            color = (0, 255, 0)  # Green by default
            if potential_fight:
                color = (0, 0, 255)  # Red during a potential fight

                # Display "Fight!" text
                cv2.putText(frame, 'Fight!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Draw a rectangle around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Increment the counter
            num_people += 1

    # Display the number of people and the frame
    cv2.putText(frame, f'No. of People: {num_people}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("People Detection", frame)

    # Update the previous frame
    prev_frame = frame.copy()

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
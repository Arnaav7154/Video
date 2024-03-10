import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier('C:\\Users\\Sachin\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\cv2\\data\\haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Loop once video is successfully loaded
while True:
    
    # Read frame
    ret, frame = cap.read()

    # If there's no frame, break the loop
    if not ret:
        break

    # Convert each frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        # Draw rectangle on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with rectangles drawn
    cv2.imshow("frame", frame)

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

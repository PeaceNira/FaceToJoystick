import cv2
import numpy as np

# Parameters for Lucas-Kanade Optical Flow, which helps in tracking face movements
lk_params = {
    'winSize': (21, 21),  # Size of the search window
    'maxLevel': 3,  # Maximum number of pyramid levels
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)  # Criteria for termination
}

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')

# Create a CLAHE object for improving the contrast of the image
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Load a pre-trained deep learning model for face detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Variables to store information about the previous frame and detected faces
prev_gray = None  # Previous frame in grayscale
prev_points = None  # Points on the face that we are tracking
previous_faces = []  # List of faces detected in the previous frame


def preprocess_frame(gray_frame):
    """Enhance lighting in the grayscale frame using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    try:
        return clahe.apply(gray_frame)
    except Exception as error:
        print("Failed to apply CLAHE. Returning original frame.")
        return gray_frame


def detect_faces_dnn(frame):
    """Detect faces in the frame using a deep learning model."""
    h, w = frame.shape[:2]  # Get the dimensions of the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    # Iterate through detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get the confidence level of detection
        if confidence > 0.5:  # Only consider detections with confidence greater than 0.5
            # Calculate bounding box for detected face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))  # Store bounding box
    return faces


def detect_faces_with_cascade(gray_frame):
    """Detect faces using Haar Cascade in the grayscale frame."""
    return face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=10)


def detect_face(frame):
    """Detect or track faces in the current frame."""
    global prev_gray, prev_points, previous_faces

    # Convert the frame to grayscale and enhance lighting
    gray_frame = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # If this is the first frame or no points to track, perform face detection
    if prev_points is None:
        return detect_faces_control(frame, gray_frame)

    # Try to track the previous points
    good_new = calculate_optical_flow(gray_frame)
    if not good_new:
        return detect_faces_control(frame, gray_frame)

    # Update tracking points and return face bounding boxes
    update_previous_points(good_new, gray_frame)
    return convert_to_bounding_boxes(good_new)


def calculate_center_point(face):
    """Calculate the center point of a detected face bounding box."""
    x, y, w, h = face
    center_x = x + w / 2
    center_y = y + h / 2
    return (center_x, center_y)


def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def detect_faces_control(frame, gray_frame):
    """Switch between deep learning and Haar Cascade for face detection."""
    faces = detect_faces_dnn(frame) or detect_faces_with_cascade(gray_frame)  # Try both methods

    # Handle the case of multiple detected faces
    if len(faces) > 1:
        if previous_faces:  # If there were faces detected in the last frame
            last_center_point = calculate_center_point(previous_faces[0])  # Center of the last detected face
            # Select the closest face to the last detected one
            faces = [min(faces, key=lambda face: distance(last_center_point, calculate_center_point(face)))]
        else:
            faces = [faces[0]]  # If no previous face, just take the first one

    # Update tracking even if no faces are detected
    if len(faces) > 0:
        update_face_tracking(faces, gray_frame)
    else:
        print("No faces detected in the current frame.")

    return faces


def calculate_optical_flow(gray_frame):
    """Track face movement by calculating optical flow."""
    global prev_gray, prev_points

    # Track points from the previous frame to the current one
    p1, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None, **lk_params)
    good_new = []

    if p1 is not None:
        # Keep only the points that are within the face bounding boxes
        for (x_new, y_new), (x_old, y_old) in zip(p1.reshape(-1, 2), prev_points.reshape(-1, 2)):
            for (x, y, w, h) in previous_faces:
                if x <= x_new <= x + w and y <= y_new <= y + h:  # Check if new point is within any detected face
                    good_new.append((x_new, y_new))  # Add good points for tracking
                    break
    return good_new


def update_face_tracking(faces, gray_frame):
    """Update the tracking points based on newly detected faces."""
    global prev_points, prev_gray, previous_faces

    # Convert detected face bounding boxes to points for tracking
    points = [[x + w / 2, y + h / 2] for x, y, w, h in faces]
    prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)  # Prepare points for optical flow
    prev_gray = gray_frame.copy()  # Save the current frame for the next calculation
    previous_faces = faces  # Update the previous face list


def update_previous_points(good_new, gray_frame):
    """Update the previous points and frame for tracking."""
    global prev_points, prev_gray
    prev_points = np.array(good_new, dtype=np.float32).reshape(-1, 1, 2)  # Prepare new points
    prev_gray = gray_frame.copy()  # Save the current frame


def convert_to_bounding_boxes(good_new):
    """Convert tracked points into bounding boxes for faces."""
    faces = []
    for (x_new, y_new) in good_new:
        w, h = 100, 100  # Default size for the bounding box
        x, y = int(x_new - w / 2), int(y_new - h / 2)  # Calculate top-left corner of the bounding box
        faces.append((x, y, w, h))  # Store the bounding box
    return faces
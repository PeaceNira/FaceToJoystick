import cv2
import numpy as np

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml'
)

# Global variables to store previous frame and face points
prev_gray = None
prev_points = None
previous_faces = []


def detect_face(frame):
    """Detect faces in the given frame.

    Args:
        frame: The input frame.

    Returns:
        List of bounding boxes around detected faces.
    """
    global prev_gray, prev_points, previous_faces

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If no previous points, use Haar cascade for face detection
    if prev_points is None:
        return detect_faces_with_cascade(frame, gray_frame)
    
    if not (good_new := calculate_optical_flow(gray_frame)):
        return expand_search_area(gray_frame)
    update_previous_points(good_new, gray_frame)
    return convert_to_bounding_boxes(good_new)


def calculate_optical_flow(gray_frame):
    """Calculate optical flow using Lucas-Kanade method.

    Args:
        gray_frame: The grayscale frame.

    Returns:
        List of new points representing the optical flow.
    """
    global prev_gray, prev_points, previous_faces

    # Calculate optical flow using Lucas-Kanade method
    p1, _, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray_frame, prev_points, None, **lk_params)
    good_new = []
    if p1 is not None:
        # Filter points that are within the bounding box of previous faces
        for (x_new, y_new), (x_old, y_old) in zip(p1.reshape(-1, 2), prev_points.reshape(-1, 2)):
            for (x, y, w, h) in previous_faces:
                if x <= x_new <= x + w and y <= y_new <= y + h:
                    good_new.append((x_new, y_new))
                    break
    return good_new


def update_previous_points(good_new, gray_frame):
    """Update previous points and gray frame.

    Args:
        good_new: List of new points representing the optical flow.
        gray_frame: The grayscale frame.
    """
    global prev_points, prev_gray

    # Update previous points and gray frame
    prev_points = np.array(good_new, dtype=np.float32).reshape(-1, 1, 2)
    prev_gray = gray_frame.copy()


def convert_to_bounding_boxes(good_new):
    """Convert new points to bounding boxes.

    Args:
        good_new: List of new points representing the optical flow.

    Returns:
        List of bounding boxes around detected faces.
    """
    # Convert points to bounding boxes
    faces = []
    for (x_new, y_new) in good_new:
        w, h = 100, 100
        x, y = int(x_new - w / 2), int(y_new - h / 2)
        faces.append((x, y, w, h))
    return faces


def expand_search_area(gray_frame):
    """Expand search area if face is not found.

    Args:
        gray_frame: The grayscale frame.

    Returns:
        List of bounding boxes around detected faces.
    """
    global previous_faces

    # Expand search area if face is not found
    search_area = 50
    for i in range(1, 5):
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.05, minNeighbors=10, minSize=(search_area*i, search_area*i))
        if len(faces) > 0:
            points = [[x + w / 2, y + h / 2] for x, y, w, h in faces]
            prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            prev_gray = gray_frame.copy()
            return faces
    return previous_faces


def detect_faces_with_cascade(frame, gray_frame):
    """Detect faces using Haar cascade.

    Args:
        frame: The input frame.
        gray_frame: The grayscale frame.

    Returns:
        List of bounding boxes around detected faces.
    """
    global prev_points, prev_gray, previous_faces

    if len(previous_faces) > 0:
        faces = previous_faces
    else:
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.05, minNeighbors=10)

    if len(faces) == 0 and len(previous_faces) > 0:
        # If no face is found and there were faces detected previously, display message
        cv2.putText(
            frame, "No face found", (20,
                                     180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )
        return previous_faces
    elif len(faces) > 0:
        # If faces are found, update previous points and gray frame
        points = [[x + w / 2, y + h / 2] for x, y, w, h in faces]
        prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        prev_gray = gray_frame.copy()
        previous_faces = faces
        return faces
    else:
        return []
import cv2
from FaceDetection import detect_face
from JoystickControl import map_face_to_joystick
import vgamepad as vg
import constants

# Initialize the gamepad
gamepad = vg.VX360Gamepad()


def display_text(frame, text, y):
    """Display text on the frame."""
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)


def display_joystick_info(frame):
    """Display joystick information on the frame."""
    # Information about joystick values, deadzone, and sensitivity
    joystick_info = [
        (f"Joystick Value (X): {constants.smoothed_joystick_value_x:.2f}", 40),
        (f"Joystick Value (Y): {constants.smoothed_joystick_value_y:.2f}", 80),
        (
            f"Joystick Deadzone (X, Y): {constants.deadzone_threshold_x:.2f}, "
            f"{constants.deadzone_threshold_y:.2f}",
            120,
        ),
        (
            f"Joystick Sensitivity (X, Y): {constants.sensitivity_x:.2f}, "
            f"{constants.sensitivity_y:.2f}",
            160,
        ),
    ]
    # Display each piece of joystick information
    for text, y in joystick_info:
        display_text(frame, text, y)


def draw_face(frame, x, y, w, h):
    """Draw the detected face on the frame."""
    # Calculate the center of the face
    face_x = x + w // 2
    face_y = y + h // 2

    # Draw a rectangle around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Apply Gaussian blur to the face region
    roi = frame[y: y + h, x: x + w]
    roi = cv2.GaussianBlur(roi, (23, 23), 30)
    frame[y: y + roi.shape[0], x: x + roi.shape[1]] = roi

    # Draw a circle at the center of the face
    cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)

    return face_x, face_y


def update_initial_face_position(face_x, face_y):
    """Update the initial position of the face."""
    if constants.initial_face_x is None or constants.initial_face_y is None:
        constants.initial_face_x = face_x
        constants.initial_face_y = face_y


def update_joystick_values(frame, face_x, face_y):
    """Update joystick values based on face position."""
    # Map face position to joystick values
    joystick_value_x, joystick_value_y = map_face_to_joystick(
        frame.shape[1],
        frame.shape[0],
        face_x,
        face_y,
        constants.initial_face_y,
        constants.initial_face_x,
        constants.deadzone_threshold_x,
        constants.deadzone_threshold_y,
        constants.sensitivity_x,
        constants.sensitivity_y,
    )

    # Smooth joystick values using exponential moving average
    constants.smoothed_joystick_value_x = (
        constants.smoothing_factor * joystick_value_x
        + (1 - constants.smoothing_factor) *
        constants.smoothed_joystick_value_x
    )
    constants.smoothed_joystick_value_y = (
        constants.smoothing_factor * joystick_value_y
        + (1 - constants.smoothing_factor) *
        constants.smoothed_joystick_value_y
    )

    # Control the gamepad with smoothed joystick values
    gamepad.right_joystick_float(
        x_value_float=constants.smoothed_joystick_value_x,
        y_value_float=constants.smoothed_joystick_value_y,
    )
    gamepad.update()


def process_frame(frame):
    """Process each frame."""
    # Rotate the frame
    if constants.rotate_camera:   
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Detect faces in the frame
    faces = detect_face(frame)

    # Display joystick information on the frame
    display_joystick_info(frame)

    # Process each detected face
    for x, y, w, h in faces:
        face_x, face_y = draw_face(frame, x, y, w, h)
        update_initial_face_position(face_x, face_y)
        update_joystick_values(frame, face_x, face_y)

    return frame

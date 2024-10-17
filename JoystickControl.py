import numpy as np


def map_face_to_joystick(
    frame_width,
    frame_height,
    face_x,
    face_y,
    initial_face_y,
    initial_face_x,
    deadzone_threshold_x,
    deadzone_threshold_y,
    sensitivity_x,
    sensitivity_y,
):
    # Ensure both face_x and face_y have valid values
    if face_x is None or face_y is None:
        print("Invalid face coordinates: face_x and face_y must have valid values.")
        return 0.0, 0.0  # Return default joystick values if face coordinates are invalid

    joystick_center_x = frame_width // 2
    joystick_center_y = frame_height // 2

    # Initialize initial face position if not set
    if initial_face_x is None or initial_face_y is None:
        initial_face_x = face_x
        initial_face_y = face_y

    # Calculate joystick values based on face movement
    joystick_value_x = (face_x - initial_face_x) / joystick_center_x * sensitivity_x
    joystick_value_y = (initial_face_y - face_y) / joystick_center_y * sensitivity_y

    # Apply deadzone logic and clip joystick values
    joystick_value_x = np.clip(joystick_value_x, -1, 1) if abs(joystick_value_x) >= deadzone_threshold_x / joystick_center_x else 0.0
    joystick_value_y = np.clip(joystick_value_y, -1, 1) if abs(joystick_value_y) >= deadzone_threshold_y / joystick_center_y else 0.0

    return joystick_value_x, joystick_value_y

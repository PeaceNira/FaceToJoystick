import numpy as np

buffer_size = 5
joystick_buffer_x = np.zeros(buffer_size)
joystick_buffer_y = np.zeros(buffer_size)
buffer_index = 0


def smooth_joystick_value(value, buffer):
    global buffer_index
    buffer[buffer_index] = value
    buffer_index = (buffer_index + 1) % len(buffer)
    return np.median(buffer)


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
    
    # Calculate joystick center 
    joystick_center_x = frame_width // 2
    joystick_center_y = frame_height // 2

    # Max value
    max_joystick_value = 1.0

    # Set initial face if not already set
    if initial_face_x is None or initial_face_y is None:
        initial_face_x = face_x
        initial_face_y = face_y

    # Calculate joystick values
    joystick_value_x = (
        (face_x - initial_face_x) / joystick_center_x * max_joystick_value * sensitivity_x
    )
    joystick_value_y = (
        (initial_face_y - face_y) / joystick_center_y * max_joystick_value * sensitivity_y
    )

    # Applys deadzone 
    if abs(joystick_value_x) < deadzone_threshold_x / joystick_center_x:
        joystick_value_x = 0.0
    if abs(joystick_value_y) < deadzone_threshold_y / joystick_center_y:
        joystick_value_y = 0.0

    # Enforces a value between -1 and 1
    joystick_value_x = max(min(joystick_value_x, 1.0), -1.0)
    joystick_value_y = max(min(joystick_value_y, 1.0), -1.0)

    return joystick_value_x, joystick_value_y

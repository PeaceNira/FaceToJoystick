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
    joystick_center_x = frame_width // 2
    joystick_center_y = frame_height // 2

    if initial_face_x is None or initial_face_y is None:
        initial_face_x = face_x
        initial_face_y = face_y

    joystick_value_x = (face_x - initial_face_x) / \
        joystick_center_x * sensitivity_x
    joystick_value_y = (initial_face_y - face_y) / \
        joystick_center_y * sensitivity_y

    joystick_value_x = np.clip(joystick_value_x, -1, 1) if abs(
        joystick_value_x) >= deadzone_threshold_x / joystick_center_x else 0.0
    joystick_value_y = np.clip(joystick_value_y, -1, 1) if abs(
        joystick_value_y) >= deadzone_threshold_y / joystick_center_y else 0.0

    return joystick_value_x, joystick_value_y

DEADZONE_THRESHOLD_X = 25
DEADZONE_THRESHOLD_Y = 25
SENSITIVITY_Y = 10
SENSITIVITY_X = 5
SMOOTHING = 0.5

def map_face_to_joystick(frame_width, frame_height, face_x, face_y, initial_face_y, initial_face_x):
    joystick_center_x = frame_width // 2
    joystick_center_y = frame_height // 2
    max_joystick_value = 1.0

    if initial_face_x is None or initial_face_y is None:
        initial_face_x = face_x
        initial_face_y = face_y

    joystick_value_x = (
        (face_x - initial_face_x) / joystick_center_x * max_joystick_value * SENSITIVITY_X
    )
    joystick_value_y = (
        (initial_face_y - face_y) / joystick_center_y * max_joystick_value * SENSITIVITY_Y
    )

    if abs(joystick_value_x) < DEADZONE_THRESHOLD_X / joystick_center_x:
        joystick_value_x = 0.0
    if abs(joystick_value_y) < DEADZONE_THRESHOLD_Y / joystick_center_y:
        joystick_value_y = 0.0

    joystick_value_x = max(min(joystick_value_x, 1.0), -1.0)
    joystick_value_y = max(min(joystick_value_y, 1.0), -1.0)

    return joystick_value_x, joystick_value_y


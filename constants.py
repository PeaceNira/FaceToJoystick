# Configuration parameters
deadzone_threshold_x = 20
deadzone_threshold_y = 20
sensitivity_y = 5
sensitivity_x = 5
smoothing_factor = 1.0
rotate_camera = False

# Initialize face tracking parameters
initial_face_x = None
initial_face_y = None

# Initialize smoothed joystick values
smoothed_joystick_value_x = 0.0
smoothed_joystick_value_y = 0.0

def on_change_deadzone_x(val):
    global deadzone_threshold_x
    deadzone_threshold_x = val

def on_change_deadzone_y(val):
    global deadzone_threshold_y
    deadzone_threshold_y = val

def on_change_sensitivity_x(val):
    global sensitivity_x
    sensitivity_x = val

def on_change_sensitivity_y(val):
    global sensitivity_y
    sensitivity_y = val
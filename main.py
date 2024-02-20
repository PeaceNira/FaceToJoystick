import cv2
import vgamepad as vg
from FaceDetection import detect_face
import JoystickControl
import numpy as np

# Initialize gamepad
gamepad = vg.VX360Gamepad()

# Configuration parameters
deadzone_threshold_x = 20
deadzone_threshold_y = 20
sensitivity_y = 5
sensitivity_x = 5
smoothing_factor = 0.2

# Initialize face tracking parameters
initial_face_x = None
initial_face_y = None

# Initialize smoothed joystick values
smoothed_joystick_value_x = 0.0
smoothed_joystick_value_y = 0.0

def process_frame(frame):
    global initial_face_x, initial_face_y, smoothed_joystick_value_x, smoothed_joystick_value_y

    # Rotate frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Detect faces
    faces = detect_face(frame)

    # Display joystick values
    display_text(frame, f"Joystick Value (X): {smoothed_joystick_value_x:.2f}", 40)
    display_text(frame, f"Joystick Value (Y): {smoothed_joystick_value_y:.2f}", 80)
    display_text(
        frame,
        f"Joystick Deadzone (X, Y): {deadzone_threshold_x:.2f}, {deadzone_threshold_y:.2f}",
        120,
    )
    display_text(
        frame,
        f"Joystick Sensitivity (X, Y): {sensitivity_x:.2f}, {sensitivity_y:.2f}",
        160,
    )

    for x, y, w, h in faces:
        face_x = x + w // 2
        face_y = y + h // 2

        # Draw face rectangle and center point
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)

        # Update initial face position if not set
        if initial_face_x is None or initial_face_y is None:
            initial_face_x = face_x
            initial_face_y = face_y

        # Map face position to joystick values
        joystick_value_x, joystick_value_y = JoystickControl.map_face_to_joystick(
            frame.shape[1],
            frame.shape[0],
            face_x,
            face_y,
            initial_face_y,
            initial_face_x,
            deadzone_threshold_x,
            deadzone_threshold_y,
            sensitivity_x,
            sensitivity_y,
        )

        # Smooth joystick values using exponential moving average
        smoothed_joystick_value_x = (
            smoothing_factor * joystick_value_x
            + (1 - smoothing_factor) * smoothed_joystick_value_x
        )
        smoothed_joystick_value_y = (
            smoothing_factor * joystick_value_y
            + (1 - smoothing_factor) * smoothed_joystick_value_y
        )

        # Control gamepad with smoothed joystick values
        gamepad.right_joystick_float(
            x_value_float=smoothed_joystick_value_x,
            y_value_float=smoothed_joystick_value_y,
        )
        gamepad.update()

    return frame


def display_text(frame, text, y):
    cv2.putText(
        frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )


def main():
    global initial_face_x
    global initial_face_y
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame = process_frame(frame)

        # Display frame
        cv2.imshow("FaceToJoystick", frame)
        cv2.setWindowProperty(
            "FaceToJoystick", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            initial_face_x = None
            initial_face_y = None
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

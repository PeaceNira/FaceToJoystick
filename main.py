import cv2
import os
import sys
import vgamepad as vg
from FaceDetection import detect_face
import JoystickControl
import numpy as np
from helpers import process_frame
import constants

# Initialize gamepad
gamepad = vg.VX360Gamepad()


def main():
    global initial_face_x, initial_face_y
    cap = cv2.VideoCapture(0)

    # Initialize trackbars
    cv2.namedWindow("FaceToJoystick")
    cv2.createTrackbar("Deadzone X", "FaceToJoystick",
                       constants.deadzone_threshold_x, 100, constants.on_change_deadzone_x)
    cv2.createTrackbar("Deadzone Y", "FaceToJoystick",
                       constants.deadzone_threshold_y, 100, constants.on_change_deadzone_y)
    cv2.createTrackbar("Sensitivity X", "FaceToJoystick",
                       constants.sensitivity_x, 20, constants.on_change_sensitivity_x)
    cv2.createTrackbar("Sensitivity Y", "FaceToJoystick",
                       constants.sensitivity_y, 20, constants.on_change_sensitivity_y)

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
            constants.initial_face_x = None
            constants.initial_face_y = None
        elif key == ord("q"):
            break
        elif key == ord("s"):
            os.execv(sys.executable, ['python'] + sys.argv)
            

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import vgamepad as vg
from FaceDetection import detect_face
from JoystickControl import map_face_to_joystick

gamepad = vg.VX360Gamepad()
initial_face_x = None
initial_face_y = None
prev_joystick_x = 0.0
prev_joystick_y = 0.0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detect_face(frame)

    for (x, y, w, h) in faces:
        face_x = x + w // 2
        face_y = y + h // 2
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
        
        if initial_face_x is None or initial_face_y is None:
            initial_face_x = face_x
            initial_face_y = face_y
        
        joystick_value_x, joystick_value_y = map_face_to_joystick(
            frame.shape[1], frame.shape[0], face_x, face_y, initial_face_y, initial_face_x
        )
        
        gamepad.right_joystick_float(x_value_float=joystick_value_x, y_value_float=joystick_value_y)
        gamepad.update()
        cv2.putText(frame, f"Joystick Value (X): {joystick_value_x:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Joystick Value (Y): {joystick_value_y:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Head Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        initial_face_x = None
        initial_face_y = None
    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

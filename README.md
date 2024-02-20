# FaceToJoystick

FaceToJoystick is a tool allowing you to use your head as a joystick, similiar to opentrack but more compatable with linux.

## Requirements

- Python 3.x
- OpenCV (`opencv-python` package)
- vgamepad library (for virtual joystick control)
- Webcam (built-in or external) for capturing facial movements

## Installation

1. Clone the repository to your local machine:

```
git clone https://github.com/PeaceNira/FaceToJoystick.git
```

2. Install the required Python packages:

```
pip install opencv-python vgamepad
```

3. Connect your webcam to your computer.

## Usage

1. Run the `main.py` script:

```
python main.py
```

2. Position your face in front of the webcam.

3. Look left, right, up, or down to control the virtual joystick.

4. Press the 'r' key to reset the initial face position if needed.

5. Press the 'q' key to exit the program.

6. The X output is Controller axis 4, and the Y output is axis 5.

## Contributing

Contributions are welcome and am hoping to build this project into a good linux alternative to opentrack. 

## License

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.

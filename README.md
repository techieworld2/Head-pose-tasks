# Head Pose Task Recorder

This project is a Python application that uses a webcam to guide a user through a series of head pose tasks (look up, down, left, right), verifies completion using face mesh detection, and saves a snapshot to a local SQLite database along with user information.

## Features

- Guides user through sequential head pose tasks.
- Uses `cvzone` and `opencv-python` for real-time face mesh detection.
- Stores user ID, name, and captured image (as base64) in a local SQLite database.
- Simple calibration for neutral head pose.
- Visual feedback and error handling in the UI.

## Requirements

- Python 3.7+
- Webcam

### Python Packages

Install dependencies using:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
opencv-python
numpy
cvzone
```

## Usage

1. Clone the repository and navigate to the project directory.
2. Ensure your webcam is connected.
3. Run the script:

   ```bash
   python corrected_face_new.py
   ```

4. Enter your ID and Name when prompted.
5. Press `c` to calibrate your neutral head pose.
6. Follow the on-screen instructions to complete each head pose task.
7. The application will save your information and a snapshot to `headpose_results.db` upon completion.

## Output

- A SQLite database file `headpose_results.db` will be created in the project directory, containing user info and the captured image.

## Notes

- The script will create the database if it does not exist.
- Press `q` to quit at any time.
- Make sure your environment has access to a webcam.

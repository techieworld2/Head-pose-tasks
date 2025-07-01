import cv2
import random
import time
import numpy as np
import sqlite3
import base64
from cvzone.FaceMeshModule import FaceMeshDetector
from collections import deque

# Get user info
user_id = input("Enter your ID: ")
user_name = input("Enter your Name: ")

# Initialize DB
def init_db():
    conn = sqlite3.connect("headpose_results.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (
        id TEXT,
        name TEXT,
        image_b64 TEXT
    )''')
    conn.commit()
    conn.close()

def save_to_db(user_id, user_name, image_b64):
    conn = sqlite3.connect("headpose_results.db")
    c = conn.cursor()
    c.execute("INSERT INTO results (id, name, image_b64) VALUES (?, ?, ?)",
              (user_id, user_name, image_b64))
    conn.commit()
    conn.close()

init_db()

# Initialize webcam and detector
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

# Sequential tasks
tasks = ["Look Up", "Look Down", "Look Left", "Look Right"]
task_index = 0
task_completed = False
stable_frame_counter = 0
wrong_attempts = 0
task_start_time = None
task_duration = 10
show_wrong_msg = False
wrong_msg_timer = 0

# Neutral pose vars
neutral_pitch = None
neutral_yaw = None
neutral_nose_position = None
calibrated = False

# Smoothing
pitch_queue = deque(maxlen=10)
yaw_queue = deque(maxlen=10)

def get_head_pose(face):
    left_jaw = face[234]
    right_jaw = face[454]
    forehead = face[10]
    chin = face[152]

    head_width = np.linalg.norm(np.array(left_jaw) - np.array(right_jaw))
    head_height = np.linalg.norm(np.array(forehead) - np.array(chin))

    nose = face[1]
    mid_x = (left_jaw[0] + right_jaw[0]) / 2
    yaw = (nose[0] - mid_x) / head_width * 100
    mid_y = (forehead[1] + chin[1]) / 2
    pitch = (nose[1] - mid_y) / head_height * 100

    return pitch, yaw, nose

def update_pose(pitch, yaw):
    pitch_queue.append(pitch)
    yaw_queue.append(yaw)
    avg_pitch = sum(pitch_queue) / len(pitch_queue)
    avg_yaw = sum(yaw_queue) / len(yaw_queue)
    return avg_pitch, avg_yaw

def is_neutral(pitch, yaw, tolerance=5):
    if neutral_pitch is None or neutral_yaw is None:
        return False
    return abs(pitch - neutral_pitch) < tolerance and abs(yaw - neutral_yaw) < tolerance

def calibrate_neutral(pitch, yaw, nose):
    global neutral_pitch, neutral_yaw, neutral_nose_position, calibrated
    neutral_pitch = pitch
    neutral_yaw = yaw
    neutral_nose_position = nose
    calibrated = True
    print(f"Calibrated: Pitch={pitch:.2f}, Yaw={yaw:.2f}")

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        raw_pitch, raw_yaw, nose = get_head_pose(face)
        avg_pitch, avg_yaw = update_pose(raw_pitch, raw_yaw)

        if calibrated:
            # Draw neutral point
            if neutral_nose_position:
                cv2.circle(img, (int(neutral_nose_position[0]), int(neutral_nose_position[1])), 8, (0, 255, 0), -1)
                cv2.putText(img, "Return to this point",
                            (int(neutral_nose_position[0]) - 50, int(neutral_nose_position[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if task_index < len(tasks):
                current_task = tasks[task_index]

                if current_task == "Look Left" and avg_yaw < -15:
                    task_completed = True
                elif current_task == "Look Right" and avg_yaw > 15:
                    task_completed = True
                elif current_task == "Look Up" and avg_pitch < -10:
                    task_completed = True
                elif current_task == "Look Down" and avg_pitch > 10:
                    task_completed = True
                elif time.time() - task_start_time > task_duration:
                    wrong_attempts += 1
                    show_wrong_msg = True
                    wrong_msg_timer = time.time()
                    print("Incorrect! Please perform the task again.")
                    task_start_time = time.time()
                    time.sleep(1)

                # Wait for return to neutral
                if task_completed and is_neutral(avg_pitch, avg_yaw):
                    print(f"{current_task} done. Moving to next task...")
                    task_index += 1
                    task_completed = False
                    show_wrong_msg = False
                    task_start_time = time.time()
                    time.sleep(1)

                # UI
                time_left = max(0, int(task_duration - (time.time() - task_start_time)))
                cv2.putText(img, f"Task: {current_task}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3)
                cv2.putText(img, f"Time Left: {time_left}s", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(img, f"Wrong Attempts: {wrong_attempts}", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if show_wrong_msg and time.time() - wrong_msg_timer < 2:
                    cv2.putText(img, "Incorrect! Perform the task again.",
                                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            else:
                # All tasks done — capture image
                _, jpeg_img = cv2.imencode('.jpg', img)
                image_bytes = jpeg_img.tobytes()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                save_to_db(user_id, user_name, image_b64)
                print("All tasks completed. Image saved to DB.")
                cv2.putText(img, "All tasks completed!", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow("Head Pose Task", img)
                cv2.waitKey(3000)
                break

        else:
            cv2.putText(img, "Press 'c' to calibrate neutral pose", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Head Pose Task", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('c') and faces:
        pitch, yaw, nose = get_head_pose(faces[0])
        calibrate_neutral(pitch, yaw, nose)
        task_start_time = time.time()

cap.release()
cv2.destroyAllWindows()

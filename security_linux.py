"""
MIT License

Copyright (c) 2025 solveditnpc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Generalised the code for import of names and face_encodings 
# Unknown faces are now saved in a folder named unknown_faces in the project directory
# Fixed the issue where time of the recognition was same for all the faces recognised 
# Follow the dictionary format in store1.csv ie "name" "image_path" if you import your own file 
import cv2
import csv
import face_recognition
import numpy as np
import datetime
import os
import subprocess
import logging
import time

# Set up logging
logging.basicConfig(
    filename='security_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a folder for unknown faces if it doesn't exist already
unknown_faces_folder = "unknown_faces" 
os.makedirs(unknown_faces_folder, exist_ok=True)

def logout_user():
    """Function to gracefully log out the current user only"""
    try:
        # Get current user
        current_user = os.getenv('USER')
        if not current_user:
            logging.error("Could not determine current user")
            return False
            
        # Get current user's display
        display = os.getenv('DISPLAY')
        if not display:
            logging.error("Could not determine display")
            return False

        logging.warning(f"Security breach detected - Gracefully logging out user: {current_user}")
        
        # First try gentle logout via gnome-session-quit
        try:
            subprocess.run(['gnome-session-quit', '--force'], timeout=5)
            return True
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            logging.warning("gnome-session-quit failed, trying alternative method")
        
        # If gnome-session-quit fails, try TERM signal
        try:
            # Get all processes for current user on current display
            ps_cmd = f"ps -u {current_user} | grep {display}"
            processes = subprocess.check_output(ps_cmd, shell=True).decode().strip().split('\n')
            
            # Send TERM signal to user's display processes
            for proc in processes:
                if proc:
                    pid = proc.split()[0]
                    try:
                        subprocess.run(['kill', '-TERM', pid])
                    except subprocess.SubprocessError:
                        continue
            
            logging.info(f"Sent termination signal to {current_user}'s processes")
            return True
            
        except Exception as e:
            logging.error(f"Failed to terminate user processes: {str(e)}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to log out user: {str(e)}")
        return False

video_capture = cv2.VideoCapture(0)

# Specify the folder location and csv file location
database_folder = "database1"
csv_file = "store1.csv"

# Store the faceencodings and names of the people in the database
known_face_encodings = []
known_face_names = []

# Read the CSV file and load face encodings
with open(csv_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_path = os.path.join(database_folder, row['image_path'])
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(row['name'])
            else:
                print(f"Warning: No face detected in image {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")

# Creating a copy of the names list
students = known_face_names.copy()

# Get the current date for the CSV filename
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Open the CSV file for writing, or create it 
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# Initialize a counter for unknown faces
unknown_face_counter = 1

#Check if the unknown faces in the webcam feed are already in the unknown folder  
def is_face_unique(face_encoding, folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            saved_image = face_recognition.load_image_file(os.path.join(folder_path, filename))
            saved_face_encodings = face_recognition.face_encodings(saved_image)
            if saved_face_encodings:
                saved_face_encoding = saved_face_encodings[0]
                matches = face_recognition.compare_faces([saved_face_encoding], face_encoding)
                if matches[0]:
                    return False
    return True

# Initialize variables for security features
unknown_face_detected_time = None
SECURITY_THRESHOLD_TIME = 3  # Seconds to wait before logout
security_warning_shown = False

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue
        
    # Convert the image from BGR color to RGB color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Only attempt face encoding if faces are found
    face_encodings = []
    if face_locations:
        try:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        except Exception as e:
            logging.error(f"Error encoding faces: {e}")
            continue

    unknown_detected = False
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        name = "Unknown"

        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distance) > 0:  # Check if there are any known faces to compare with
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                logging.info(f"Authorized user detected: {name}")
                unknown_face_detected_time = None
                security_warning_shown = False
         
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        if name == "Unknown":
            unknown_detected = True
            # Save unknown face if unique
            if is_face_unique(face_encoding, unknown_faces_folder):
                face_image = frame[top:bottom, left:right]
                face_filename = f"{unknown_faces_folder}/unknown_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(face_filename, face_image)
                logging.warning(f"Unknown face detected and saved: {face_filename}")
                
            # Start timer for security response if not already started
            if unknown_face_detected_time is None:
                unknown_face_detected_time = time.time()
                logging.warning("Unknown face detected - Starting security countdown")

        # Display name with red background for unknown faces
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        if name in students:
            students.remove(name)
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            lnwriter.writerow([name, current_time])

    # Security response for unknown faces
    if unknown_detected and unknown_face_detected_time is not None:
        time_elapsed = time.time() - unknown_face_detected_time
        if time_elapsed >= SECURITY_THRESHOLD_TIME:
            # Display warning message on frame
            warning_text = "SECURITY ALERT - Unauthorized User Detected"
            cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            
            if not security_warning_shown:
                logging.critical("Security threshold exceeded - Initiating logout")
                logout_user()
                security_warning_shown = True
    
    cv2.imshow('Security Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
logging.info("Security system shutdown")
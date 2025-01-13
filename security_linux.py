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

import face_recognition
import os
import logging
import numpy as np
from typing import Optional, Tuple, List
import cv2
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_image_encodings(image_path: str) -> Tuple[Optional[List[np.ndarray]], str]:
    try:
        if not os.path.exists(image_path):
            return None, f"Image file not found: {image_path}"
            
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            return None, f"No faces found in image: {image_path}"
            
        return face_encodings, ""
        
    except Exception as e:
        return None, f"Error processing image {image_path}: {str(e)}"

def compare_faces(input_image_path: str, database_folder: str, tolerance: float = 0.6) -> bool:
    if not os.path.exists(database_folder):
        logging.error(f"Database folder not found: {database_folder}")
        return False
        
    input_encodings, error = load_image_encodings(input_image_path)
    if error:
        logging.error(error)
        return False
        
    for filename in os.listdir(database_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            db_image_path = os.path.join(database_folder, filename)
            
            db_encodings, error = load_image_encodings(db_image_path)
            if error:
                logging.warning(error)
                continue
                
            for input_encoding in input_encodings:
                matches = face_recognition.compare_faces(db_encodings, input_encoding, tolerance=tolerance)
                if True in matches:
                    logging.info(f"Face match found with database image: {filename}")
                    return True
                    
    logging.info("No matching faces found in database")
    return False

def capture_and_compare(database_folder: str = "database1", tolerance: float = 0.6):
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logging.error("Could not open webcam")
        return

    last_check_time = time.time()
    CHECK_INTERVAL = 5.0  # Check every 3 seconds
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Failed to capture frame")
                continue

            current_time = time.time()
            if current_time - last_check_time >= CHECK_INTERVAL:
                logging.info("Capturing and processing frame...")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                face_encodings = face_recognition.face_encodings(rgb_frame)
                
                if face_encodings:
                    match_found = False
                    for filename in os.listdir(database_folder):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            db_image_path = os.path.join(database_folder, filename)
                            db_encodings, error = load_image_encodings(db_image_path)
                            
                            if error:
                                continue
                                
                            for face_encoding in face_encodings:
                                matches = face_recognition.compare_faces(db_encodings, face_encoding, tolerance=tolerance)
                                if True in matches:
                                    logging.info(f"Face match found with database image: {filename}")
                                    match_found = True
                                    break
                            
                            if match_found:
                                break
                    
                    if not match_found:
                        logging.info("No matching faces found in database")
                else:
                    logging.info("No faces detected in current frame")
                
                last_check_time = current_time

            time_to_next = max(0, CHECK_INTERVAL - (current_time - last_check_time))
            countdown_text = f"Next check in: {time_to_next:.1f}s"
            cv2.putText(frame, countdown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"Error in capture_and_compare: {str(e)}")
    
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_compare("database1")
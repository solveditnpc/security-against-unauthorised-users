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
import cv2
import numpy as np
import os
import logging
import subprocess
import datetime
from typing import List, Tuple, Optional

logging.basicConfig(
    filename='security_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SecurityCamera:
    def __init__(self, database_path: str = "database1", tolerance: float = 0.6):
        self.database_path = database_path
        self.tolerance = tolerance
        self.unknown_faces_dir = "unknown_faces"
        self.check_interval = 5.0
        
        os.makedirs(self.unknown_faces_dir, exist_ok=True)
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.known_faces, self.known_names = self._load_known_faces()
        if not self.known_faces:
            raise RuntimeError("No known faces found in database")

    def _load_known_faces(self) -> Tuple[List[np.ndarray], List[str]]:
        known_faces = []
        known_names = []
        
        for filename in os.listdir(self.database_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                image_path = os.path.join(self.database_path, filename)
                face_image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(face_image)
                
                if encodings:
                    known_faces.append(encodings[0])
                    known_names.append(os.path.splitext(filename)[0])
                else:
                    logging.warning(f"No face found in {filename}")
                    
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                
        return known_faces, known_names

    def _save_unknown_face(self, face_image: np.ndarray) -> None:
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.unknown_faces_dir}/unknown_{timestamp}.jpg"
            cv2.imwrite(filename, face_image)
            logging.info(f"Saved unknown face: {filename}")
        except Exception as e:
            logging.error(f"Failed to save unknown face: {str(e)}")

    def _is_face_unique(self, face_encoding: np.ndarray) -> bool:
        for filename in os.listdir(self.unknown_faces_dir):
            if not filename.endswith('.jpg'):
                continue
                
            try:
                saved_image = face_recognition.load_image_file(
                    os.path.join(self.unknown_faces_dir, filename)
                )
                saved_encodings = face_recognition.face_encodings(saved_image)
                
                if saved_encodings and face_recognition.compare_faces(
                    [saved_encodings[0]], face_encoding)[0]:
                    return False
            except Exception as e:
                logging.error(f"Error checking face uniqueness: {str(e)}")
                
        return True

    def _process_frame(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]], List[str]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        authorized_found = False
        names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_faces, face_encoding, tolerance=self.tolerance
            )
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]
                authorized_found = True
                logging.info(f"Authorized user detected: {name}")
            elif self._is_face_unique(face_encoding):
                face_location = face_locations[len(names)]
                face_img = frame[
                    face_location[0]:face_location[2],
                    face_location[3]:face_location[1]
                ]
                self._save_unknown_face(face_img)
                
            names.append(name)
            
        return authorized_found, face_locations, names

    def run(self):
        last_check_time = datetime.datetime.now()
        previous_unauthorized = False
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logging.error("Failed to capture frame")
                    continue

                current_time = datetime.datetime.now()
                time_diff = (current_time - last_check_time).total_seconds()
                
                if time_diff >= self.check_interval:
                    authorized_found, face_locations, names = self._process_frame(frame)
                    
                    for (top, right, bottom, left), name in zip(face_locations, names):
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, name, (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    
                    if not authorized_found and face_locations:
                        if previous_unauthorized:
                            logging.warning("Unauthorized face detected in consecutive frames")
                            self.logout()
                            return
                        previous_unauthorized = True
                        warning = "UNAUTHORIZED ACCESS! Will logout if unauthorized in next frame"
                        cv2.putText(frame, warning, (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        previous_unauthorized = False
                        next_check = self.check_interval - time_diff
                        status = f"Next check in: {next_check:.1f}s"
                        cv2.putText(frame, status, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    last_check_time = current_time
                
                cv2.imshow('Security Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logging.error(f"Camera error: {str(e)}")
        finally:
            self.cleanup()

    def logout(self):
        try:
            current_user = os.getenv('USER')
            display = os.getenv('DISPLAY')
            
            if not current_user or not display:
                logging.error("Could not determine current user or display")
                return

            logging.warning(f"Force logging out current session for user: {current_user}")
            
            try:
                ps_cmd = f"ps -u {current_user} | grep {display}"
                processes = subprocess.check_output(ps_cmd, shell=True).decode().split('\n')
                
                for proc in processes:
                    if proc.strip():
                        try:
                            pid = proc.split()[0]
                            subprocess.run(['kill', '-9', pid])
                        except (subprocess.SubprocessError, IndexError):
                            continue
                
                logging.info(f"Forcefully terminated display session for user: {current_user}")
            except Exception as e:
                logging.error(f"Failed to terminate session processes: {str(e)}")
                
        except Exception as e:
            logging.error(f"Logout failed: {str(e)}")

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        camera = SecurityCamera("database1")
        camera.run()
    except Exception as e:
        logging.error(f"Failed to start security camera: {str(e)}")
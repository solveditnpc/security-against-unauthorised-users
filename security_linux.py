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
from pynput import keyboard
from threading import Lock

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
        self.show_camera = True
        self.warning_active = False
        self.warning_start_time = None
        self.window_name = 'Security Camera'
        self.exit_keys = {'q': False, 'u': False, 'i': False, 't': False}
        self.key_lock = Lock()
        self.keyboard_listener = None
        
        os.makedirs(self.unknown_faces_dir, exist_ok=True)
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.known_faces, self.known_names = self._load_known_faces()
        if not self.known_faces:
            raise RuntimeError("No known faces found in database")
            
        cv2.namedWindow(self.window_name)
        
        self._setup_keyboard_listener()

    def _setup_keyboard_listener(self):
        def on_press(key):
            try:
                char = key.char
                if char in self.exit_keys:
                    with self.key_lock:
                        self.exit_keys[char] = True
            except AttributeError:
                pass
                
        def on_release(key):
            try:
                char = key.char
                if char in self.exit_keys:
                    with self.key_lock:
                        self.exit_keys[char] = False
            except AttributeError:
                pass
        
        self.keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        self.keyboard_listener.start()

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

    def _save_unknown_face(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> None:
        try:
            top, right, bottom, left = face_location
            
            height, width = frame.shape[:2]
            padding = int(min(height, width) * 0.2)
            
            pad_top = max(0, top - padding)
            pad_bottom = min(height, bottom + padding)
            pad_left = max(0, left - padding)
            pad_right = min(width, right + padding)
            
            face_image = frame[pad_top:pad_bottom, pad_left:pad_right]
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.unknown_faces_dir}/unknown_{timestamp}.jpg"
            cv2.imwrite(filename, face_image)
            
            context_size = face_image.shape[:2]
            logging.info(f"Saved unknown face with context: {filename}, context size: {context_size}")
            
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

    def _check_exit_condition(self) -> bool:
        with self.key_lock:
            return all(self.exit_keys.values())

    def _process_frame(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]], List[str]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        authorized_found = False
        names = []
        
        for i, face_encoding in enumerate(face_encodings):
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
                self._save_unknown_face(frame, face_locations[i])
                
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
                
                if self._check_exit_condition():
                    logging.info("Exit sequence detected (QUIT)")
                    break
                
                if time_diff >= self.check_interval:
                    authorized_found, face_locations, names = self._process_frame(frame)
                    
                    for (top, right, bottom, left), name in zip(face_locations, names):
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, name, (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    
                    if not authorized_found and face_locations:
                        if not self.warning_active:
                            self.warning_active = True
                            self.warning_start_time = current_time
                        
                        if previous_unauthorized:
                            logging.warning("Unauthorized face detected in consecutive frames")
                            self.logout()
                            return
                        previous_unauthorized = True
                    else:
                        if authorized_found:
                            self.warning_active = False
                            self.warning_start_time = None
                        previous_unauthorized = False
                    
                    last_check_time = current_time

                if self.warning_active:
                    warning = "UNAUTHORIZED ACCESS! Will logout if unauthorized in next frame"
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 40), (frame.shape[1], 90), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    cv2.putText(frame, warning, (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    if self.warning_start_time:
                        duration = (current_time - self.warning_start_time).total_seconds()
                        duration_text = f"Warning active for: {duration:.1f}s"
                        cv2.putText(frame, duration_text, (10, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    next_check = self.check_interval - time_diff
                    status = f"Next check in: {next_check:.1f}s"
                    cv2.putText(frame, status, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if self.show_camera:
                    try:
                        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                            cv2.namedWindow(self.window_name)
                        cv2.imshow(self.window_name, frame)
                    except:
                        cv2.namedWindow(self.window_name)
                        cv2.imshow(self.window_name, frame)
                else:
                    try:
                        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                            cv2.destroyWindow(self.window_name)
                    except:
                        pass
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('h'):
                    self.show_camera = not self.show_camera
                    
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
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        camera = SecurityCamera("database1")
        camera.run()
    except Exception as e:
        logging.error(f"Failed to start security camera: {str(e)}")
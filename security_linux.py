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
import json
from typing import List, Tuple, Optional
from pynput import keyboard
from threading import Lock

logging.basicConfig(
    filename='security_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SecurityCamera:
    def __init__(self, database_path: str = "database1", tolerance: float = 0.6, max_width: int = 640):
        self.database_path = database_path
        self.tolerance = tolerance
        self.max_width = max_width
        self.unknown_faces_dir = "unknown_faces"
        self.check_interval = 10.0  # seconds between checks
        self.show_camera = True  # Toggle for camera feed visibility
        self.warning_active = False  # Track if warning should be displayed
        self.warning_start_time = None  # Track when warning started
        self.window_name = 'Security Camera'  # Name of the camera window
        self.exit_keys = {'q': False, 'u': False, 'i': False, 't': False}
        self.key_lock = Lock()
        self.keyboard_listener = None
        
        # CPU monitoring thresholds and stats
        self.cpu_high_threshold = 80  # CPU % threshold for high load
        self.cpu_medium_threshold = 60  # CPU % threshold for medium load
        self.frame_skip = 1  # Dynamic frame skip value
        self.frame_counter = 0  # Counter for frame skipping
        self.last_cpu_time = None  # Last CPU time reading
        self.last_idle_time = None  # Last CPU idle time reading
        self.cpu_history = []  # Store recent CPU readings
        self.history_size = 5  # Number of readings to keep
        self.min_frame_skip = 1  # Minimum frame skip
        self.max_frame_skip = 10  # Maximum frame skip
        self.stats_file = "cpu_stats.json"  # File to store CPU statistics
        self.stats_update_interval = 1.0  # Update stats every second
        self.last_stats_update = datetime.datetime.now()
        
        self._init_cpu_stats()
        
        os.makedirs(self.unknown_faces_dir, exist_ok=True)
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.known_faces, self.known_names = self._load_known_faces()
        if not self.known_faces:
            raise RuntimeError("No known faces found in database")
            
        cv2.namedWindow(self.window_name)
        
        self._setup_keyboard_listener()

    def _init_cpu_stats(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    self.cpu_stats = json.load(f)
            except json.JSONDecodeError:
                self.cpu_stats = {"readings": []}
        else:
            self.cpu_stats = {"readings": []}

    def _update_cpu_stats(self, cpu_usage: float):
        current_time = datetime.datetime.now()
        
        if (current_time - self.last_stats_update).total_seconds() < self.stats_update_interval:
            return
            
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else cpu_usage
        
        reading = {
            "timestamp": current_time.isoformat(),
            "current_cpu": round(cpu_usage, 2),
            "average_cpu": round(avg_cpu, 2),
            "frame_skip": self.frame_skip
        }
        
        self.cpu_stats["readings"].append(reading)
        
        if len(self.cpu_stats["readings"]) > 1000:
            self.cpu_stats["readings"] = self.cpu_stats["readings"][-1000:]
        
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.cpu_stats, f, indent=2)
            self.last_stats_update = current_time
        except Exception as e:
            logging.error(f"Failed to update CPU stats: {str(e)}")

    def _get_cpu_usage(self) -> float:
        try:
            if self.last_cpu_time is None:
                with open('/proc/stat', 'r') as f:
                    cpu = f.readline().split()[1:]
                    idle_time = float(cpu[3])
                    total_time = sum(float(x) for x in cpu)
                    self.last_idle_time = idle_time
                    self.last_cpu_time = total_time
                return 0.0
                
            with open('/proc/stat', 'r') as f:
                cpu = f.readline().split()[1:]
                idle_time = float(cpu[3])
                total_time = sum(float(x) for x in cpu)
                
            idle_diff = idle_time - self.last_idle_time
            total_diff = total_time - self.last_cpu_time
            
            self.last_idle_time = idle_time
            self.last_cpu_time = total_time
            
            if total_diff == 0:
                return 0.0
                
            cpu_usage = 100.0 * (1.0 - idle_diff/total_diff)
            return max(0.0, min(100.0, cpu_usage))  
            
        except Exception as e:
            logging.error(f"Error reading CPU usage: {str(e)}")
            return 0.0

    def _adjust_frame_skip(self, cpu_usage: float):
        self.cpu_history.append(cpu_usage)
        if len(self.cpu_history) > self.history_size:
            self.cpu_history.pop(0)
            
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        
        if avg_cpu > self.cpu_high_threshold:
            self.frame_skip = min(self.frame_skip + 1, self.max_frame_skip)
        elif avg_cpu < self.cpu_medium_threshold:
            self.frame_skip = max(self.frame_skip - 1, self.min_frame_skip)
        elif avg_cpu > self.cpu_medium_threshold:
            if cpu_usage > avg_cpu:
                self.frame_skip = min(self.frame_skip + 1, self.max_frame_skip)
            else:
                self.frame_skip = max(self.frame_skip - 1, self.min_frame_skip)
        
        self._update_cpu_stats(cpu_usage)
        
        logging.debug(f"CPU: {cpu_usage:.1f}%, Avg: {avg_cpu:.1f}%, Skip: {self.frame_skip}")

    def _adjust_check_interval(self, cpu_usage: float):
        if cpu_usage > self.cpu_high_threshold:
            self.check_interval = min(10.0, self.check_interval * 1.5)
        elif cpu_usage < self.cpu_medium_threshold:
            self.check_interval = max(5.0, self.check_interval * 0.8)

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

    def _scale_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        if width <= self.max_width:
            return frame, 1.0
            
        scale_factor = self.max_width / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled_frame = cv2.resize(frame, (new_width, new_height))
        return scaled_frame, scale_factor

    def _scale_coordinates(self, coordinates: Tuple[int, int, int, int], scale_factor: float) -> Tuple[int, int, int, int]:
        top, right, bottom, left = coordinates
        return (
            int(top / scale_factor),
            int(right / scale_factor),
            int(bottom / scale_factor),
            int(left / scale_factor)
        )

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
        scaled_frame, scale_factor = self._scale_frame(frame)
        
        rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        authorized_found = False
        names = []
        scaled_locations = []
        
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
                scaled_loc = self._scale_coordinates(face_locations[i], scale_factor)
                self._save_unknown_face(frame, scaled_loc)
                
            names.append(name)
            scaled_locations.append(self._scale_coordinates(face_locations[i], scale_factor))
            
        return authorized_found, scaled_locations, names

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
                
                cpu_usage = self._get_cpu_usage()
                self._adjust_frame_skip(cpu_usage)
                self.frame_counter = (self.frame_counter + 1) % self.frame_skip
                
                if time_diff >= self.check_interval and self.frame_counter == 0:
                    authorized_found, face_locations, names = self._process_frame(frame)
                    
                    self._adjust_check_interval(cpu_usage)
                    
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
                        previous_unauthorized = False
                    
                    last_check_time = current_time

                if self.warning_active and self.warning_start_time:
                    warning_duration = (current_time - self.warning_start_time).total_seconds()
                    if warning_duration >= self.check_interval:
                        self.warning_active = False
                        self.warning_start_time = None

                if self.warning_active:
                    warning = "UNAUTHORIZED ACCESS! Will logout if unauthorized in next frame"
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 40), (frame.shape[1], 90), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    cv2.putText(frame, warning, (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    if self.warning_start_time:
                        remaining_time = self.check_interval - (current_time - self.warning_start_time).total_seconds()
                        if remaining_time > 0:
                            duration_text = f"Warning expires in: {remaining_time:.1f}s"
                            cv2.putText(frame, duration_text, (10, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    next_check = self.check_interval - time_diff
                    avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else cpu_usage
                    status = f"Next check in: {next_check:.1f}s | CPU: {cpu_usage:.1f}% (Avg: {avg_cpu:.1f}%) | Skip: {self.frame_skip}"
                    cv2.putText(frame, status, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
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
        
        if hasattr(self, 'cpu_stats'):
            try:
                with open(self.stats_file, 'w') as f:
                    json.dump(self.cpu_stats, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save final CPU stats: {str(e)}")

if __name__ == "__main__":
    try:
        camera = SecurityCamera("database1")
        camera.run()
    except Exception as e:
        logging.error(f"Failed to start security camera: {str(e)}")
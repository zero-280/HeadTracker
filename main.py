import cv2
import numpy as np
import os
import urllib.request
from collections import deque
import threading
from queue import Queue
import time
import random

# --- CONSTANTS ---

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 60

# Crosshair properties
CROSSHAIR_COLOR = (0, 255, 0)
CROSSHAIR_COLOR_LOCKED = (0, 0, 255)  # Red when target locked
CROSSHAIR_SIZE = 25
CROSSHAIR_THICKNESS = 2

# Target outline (ellipse for face)
TARGET_ELLIPSE_COLOR = (0, 0, 255)
TARGET_ELLIPSE_THICKNESS = 2

# Crosshair movement speed
AIM_SMOOTHING = 0.35  # Increased for faster response

# Stabilization with Kalman Filter
USE_KALMAN = True

# History for additional smoothing
POSITION_HISTORY_SIZE = 3  # Reduced for faster response

# --- SHOOTING CONSTANTS ---
CONFIDENCE_THRESHOLD = 0.70  # How accurate the aim must be (0-1) - LOWERED
LOCK_TIME_THRESHOLD = 0.08   # Seconds target must be locked - MUCH FASTER
SHOT_COOLDOWN = 0.25         # Seconds between shots - FASTER FIRE RATE
MAX_DISTANCE_FOR_LOCK = 25   # Maximum pixel distance for "Lock" - INCREASED

# Recoil effect
RECOIL_STRENGTH = 6
RECOIL_DECAY = 0.8

# Hit marker effect
HIT_MARKER_DURATION = 0.25   # Seconds
HIT_MARKER_SIZE = 40

# Filename and URL for face detection classifier
CASCADE_FILE_NAME = 'haarcascade_frontalface_alt2.xml'
CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/' + CASCADE_FILE_NAME

# --- CLASSES ---

class KalmanFilter2D:
    """Kalman filter for 2D position tracking for stabilization."""
    
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
    def update(self, x, y):
        """Updates the filter with a new measurement."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return prediction[0][0], prediction[1][0]

class FaceDetectionThread(threading.Thread):
    """Separate thread for face detection for performance improvement."""
    
    def __init__(self, face_cascade):
        super().__init__()
        self.face_cascade = face_cascade
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.running = True
        self.daemon = True
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        
    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Prepare image for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Histogram equalization for better detection
                gray_optimized = self.clahe.apply(gray)
                
                # Multiple detection passes with different parameters
                # for better detection in different lighting conditions
                faces = self.face_cascade.detectMultiScale(
                    gray_optimized,
                    scaleFactor=1.05,  # Smaller value = better detection
                    minNeighbors=4,    # Less strict for more detections
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # If no faces found, try with different parameters
                if len(faces) == 0:
                    faces = self.face_cascade.detectMultiScale(
                        gray_optimized,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(40, 40)
                    )
                
                # Put result in queue
                if not self.result_queue.full():
                    self.result_queue.put(faces)
                    
    def stop(self):
        self.running = False

class ShotSystem:
    """Manages the shooting system with confidence calculation."""
    
    def __init__(self):
        self.lock_start_time = None
        self.last_shot_time = 0
        self.is_locked = False
        self.shots_fired = 0
        self.hits = 0
        self.recoil_x = 0
        self.recoil_y = 0
        self.hit_markers = []  # List of (x, y, timestamp)
        
    def calculate_confidence(self, aim_x, aim_y, target_x, target_y):
        """Calculates target confidence based on distance."""
        distance = np.sqrt((aim_x - target_x)**2 + (aim_y - target_y)**2)
        
        if distance > MAX_DISTANCE_FOR_LOCK * 2:
            return 0.0
        
        # Closer = higher confidence
        max_dist = MAX_DISTANCE_FOR_LOCK * 2
        confidence = 1.0 - (distance / max_dist)
        return max(0.0, min(1.0, confidence))
    
    def update_lock(self, confidence, current_time, target_exists):
        """Updates the lock status."""
        # Lock can only exist if target exists
        if not target_exists:
            self.lock_start_time = None
            self.is_locked = False
            return False
            
        if confidence >= CONFIDENCE_THRESHOLD:
            if self.lock_start_time is None:
                self.lock_start_time = current_time
            
            lock_duration = current_time - self.lock_start_time
            if lock_duration >= LOCK_TIME_THRESHOLD:
                self.is_locked = True
            else:
                self.is_locked = False
        else:
            self.lock_start_time = None
            self.is_locked = False
        
        return self.is_locked
    
    def can_shoot(self, current_time):
        """Checks if shooting is possible."""
        return (self.is_locked and 
                current_time - self.last_shot_time >= SHOT_COOLDOWN)
    
    def shoot(self, aim_x, aim_y, target_x, target_y, target_exists, current_time):
        """Executes a shot."""
        self.last_shot_time = current_time
        self.shots_fired += 1
        
        # Calculate if hit - only if target still exists!
        distance = np.sqrt((aim_x - target_x)**2 + (aim_y - target_y)**2)
        is_hit = (distance <= MAX_DISTANCE_FOR_LOCK) and target_exists
        
        if is_hit:
            self.hits += 1
            self.hit_markers.append((int(target_x), int(target_y), current_time))
        
        # Generate recoil effect
        self.recoil_x = random.uniform(-RECOIL_STRENGTH, RECOIL_STRENGTH)
        self.recoil_y = random.uniform(-RECOIL_STRENGTH * 0.5, RECOIL_STRENGTH * 1.5)
        
        return is_hit
    
    def update_recoil(self):
        """Updates and reduces recoil."""
        self.recoil_x *= RECOIL_DECAY
        self.recoil_y *= RECOIL_DECAY
        
        if abs(self.recoil_x) < 0.1:
            self.recoil_x = 0
        if abs(self.recoil_y) < 0.1:
            self.recoil_y = 0
    
    def get_active_hit_markers(self, current_time):
        """Returns active hit markers."""
        self.hit_markers = [(x, y, t) for x, y, t in self.hit_markers 
                           if current_time - t <= HIT_MARKER_DURATION]
        return self.hit_markers
    
    def get_accuracy(self):
        """Calculates hit rate."""
        if self.shots_fired == 0:
            return 0.0
        return (self.hits / self.shots_fired) * 100

# --- FUNCTIONS ---

def download_cascade_file():
    """Downloads the cascade file if it doesn't exist."""
    if not os.path.exists(CASCADE_FILE_NAME):
        print(f"Downloading '{CASCADE_FILE_NAME}'...")
        try:
            urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE_NAME)
            print("Download successful.")
        except Exception as e:
            print(f"Download error: {e}")
            print("Please ensure you have an internet connection and try again.")
            return False
    return True

def draw_crosshair(frame, x, y, is_locked=False, confidence=0.0):
    """Draws a detailed crosshair with lock indicator."""
    x, y = int(x), int(y)
    
    # Color based on lock status
    color = CROSSHAIR_COLOR_LOCKED if is_locked else CROSSHAIR_COLOR
    
    # Main crosshair
    cv2.line(frame, (x - CROSSHAIR_SIZE, y), (x + CROSSHAIR_SIZE, y), color, CROSSHAIR_THICKNESS)
    cv2.line(frame, (x, y - CROSSHAIR_SIZE), (x, y + CROSSHAIR_SIZE), color, CROSSHAIR_THICKNESS)
    cv2.circle(frame, (x, y), 5, color, 1)
    
    # Outer circle with confidence indicator
    outer_radius = CROSSHAIR_SIZE + 5
    cv2.circle(frame, (x, y), outer_radius, color, 1)
    
    # Confidence arc (shows how close to lock)
    if confidence > 0:
        angle = int(360 * confidence)
        cv2.ellipse(frame, (x, y), (outer_radius + 3, outer_radius + 3), 
                   -90, 0, angle, color, 2)
    
    # Lock indicator corners
    if is_locked:
        corner_size = 15
        # Top-left
        cv2.line(frame, (x - outer_radius - 10, y - outer_radius - 10),
                (x - outer_radius - 10 + corner_size, y - outer_radius - 10), color, 2)
        cv2.line(frame, (x - outer_radius - 10, y - outer_radius - 10),
                (x - outer_radius - 10, y - outer_radius - 10 + corner_size), color, 2)
        
        # Top-right
        cv2.line(frame, (x + outer_radius + 10, y - outer_radius - 10),
                (x + outer_radius + 10 - corner_size, y - outer_radius - 10), color, 2)
        cv2.line(frame, (x + outer_radius + 10, y - outer_radius - 10),
                (x + outer_radius + 10, y - outer_radius - 10 + corner_size), color, 2)
        
        # Bottom-left
        cv2.line(frame, (x - outer_radius - 10, y + outer_radius + 10),
                (x - outer_radius - 10 + corner_size, y + outer_radius + 10), color, 2)
        cv2.line(frame, (x - outer_radius - 10, y + outer_radius + 10),
                (x - outer_radius - 10, y + outer_radius + 10 - corner_size), color, 2)
        
        # Bottom-right
        cv2.line(frame, (x + outer_radius + 10, y + outer_radius + 10),
                (x + outer_radius + 10 - corner_size, y + outer_radius + 10), color, 2)
        cv2.line(frame, (x + outer_radius + 10, y + outer_radius + 10),
                (x + outer_radius + 10, y + outer_radius + 10 - corner_size), color, 2)

def draw_hit_marker(frame, x, y, alpha=1.0):
    """Draws a hit marker."""
    color = (0, 255, 255)  # Yellow
    size = int(HIT_MARKER_SIZE * alpha)
    thickness = max(1, int(3 * alpha))
    
    # X-shaped marker
    cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(frame, (x + size, y - size), (x - size, y + size), color, thickness)
    cv2.circle(frame, (x, y), int(size * 0.7), color, thickness)

def draw_muzzle_flash(frame, x, y):
    """Draws a muzzle flash effect."""
    flash_size = 30
    overlay = frame.copy()
    cv2.circle(overlay, (int(x), int(y)), flash_size, (200, 200, 255), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

def smooth_position(history, new_x, new_y):
    """Smooths position with moving average."""
    history.append((new_x, new_y))
    avg_x = sum(pos[0] for pos in history) / len(history)
    avg_y = sum(pos[1] for pos in history) / len(history)
    return avg_x, avg_y

# --- MAIN SCRIPT ---

def main():
    # 1. Download/load classifier file
    if not download_cascade_file():
        return
        
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE_NAME)
    if face_cascade.empty():
        print("Error: Cascade classifier could not be loaded.")
        return

    # 2. Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Optimized camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for lower latency
    
    # Disable auto-exposure and auto-focus for more consistent detection
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Initialization
    aim_x, aim_y = FRAME_WIDTH / 2, FRAME_HEIGHT / 2
    target_x, target_y = FRAME_WIDTH / 2, FRAME_HEIGHT / 2
    
    # Kalman filter for stabilization
    kalman_filter = KalmanFilter2D() if USE_KALMAN else None
    
    # Position history for additional smoothing
    position_history = deque(maxlen=POSITION_HISTORY_SIZE)
    position_history.append((FRAME_WIDTH / 2, FRAME_HEIGHT / 2))
    
    # Start detection thread
    detection_thread = FaceDetectionThread(face_cascade)
    detection_thread.start()
    
    # Initialize shot system
    shot_system = ShotSystem()
    
    # Tracking variables
    last_faces = []
    face_lost_frames = 0
    MAX_FACE_LOST_FRAMES = 15  # How long the last face is "remembered"
    
    # FPS tracking
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    current_fps = 0
    
    # Flash effect
    muzzle_flash_time = 0
    FLASH_DURATION = 0.05

    print("=== AI AIMING SIMULATOR ===")
    print("Press 'q' to quit...")
    print("System shoots automatically when target is locked!")

    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for more natural display
        frame = cv2.flip(frame, 1)

        # Send frame for detection (only if queue not full)
        if detection_thread.frame_queue.qsize() < 1:
            detection_thread.frame_queue.put(frame.copy())

        # Get detection result
        if not detection_thread.result_queue.empty():
            faces = detection_thread.result_queue.get()
            if len(faces) > 0:
                last_faces = faces
                face_lost_frames = 0
            else:
                face_lost_frames += 1
        else:
            # If no new results, count frames without face
            face_lost_frames += 1

        # Use last known face
        target_exists = False  # Flag if target is present
        target_is_current = False  # Flag if target is current (not from memory)
        
        if face_lost_frames < MAX_FACE_LOST_FRAMES and len(last_faces) > 0:
            faces = last_faces
            # Only mark as "currently existing" if detected in this frame
            target_is_current = (face_lost_frames == 0)
            # Target "exists" even if recently seen (for tracking)
            target_exists = True
        else:
            faces = []
            target_exists = False
            target_is_current = False

        if len(faces) > 0:
            # Select the largest face (with the largest area w*h)
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])

            # *** PRECISE HEADSHOT LOGIC ***
            raw_target_x = x + w // 2
            raw_target_y = y + int(h * 0.35)
            
            # Apply Kalman filter for stability
            if USE_KALMAN and kalman_filter:
                target_x, target_y = kalman_filter.update(raw_target_x, raw_target_y)
            else:
                target_x, target_y = raw_target_x, raw_target_y
            
            # Additional smoothing with history
            target_x, target_y = smooth_position(position_history, target_x, target_y)

            # Draw ellipse around detected face
            center = (x + w // 2, y + h // 2)
            axes = (int(w / 1.9), int(h / 1.4))
            cv2.ellipse(frame, center, axes, 0, 0, 360, TARGET_ELLIPSE_COLOR, TARGET_ELLIPSE_THICKNESS)
            
            # Show target area
            target_point = (int(raw_target_x), int(raw_target_y))
            cv2.circle(frame, target_point, 3, (255, 0, 0), -1)
        else:
            # If no face found, return to center
            target_x = FRAME_WIDTH / 2
            target_y = FRAME_HEIGHT / 2

        # Calculate confidence
        confidence = shot_system.calculate_confidence(aim_x, aim_y, target_x, target_y)
        is_locked = shot_system.update_lock(confidence, current_time, target_exists)
        
        # Automatic shooting - only when target is CURRENT!
        if shot_system.can_shoot(current_time) and target_is_current:
            is_hit = shot_system.shoot(aim_x, aim_y, target_x, target_y, target_is_current, current_time)
            muzzle_flash_time = current_time
            
            hit_status = 'HIT!' if is_hit else 'MISS (target moved/too far)'
            print(f"SHOT #{shot_system.shots_fired} - {hit_status} - Accuracy: {shot_system.get_accuracy():.1f}%")
        
        # Apply recoil
        shot_system.update_recoil()
        target_x += shot_system.recoil_x
        target_y += shot_system.recoil_y
        
        # Move crosshair smoothly
        aim_x += (target_x - aim_x) * AIM_SMOOTHING
        aim_y += (target_y - aim_y) * AIM_SMOOTHING

        # Muzzle flash effect
        if current_time - muzzle_flash_time < FLASH_DURATION:
            draw_muzzle_flash(frame, aim_x, aim_y)
        
        # Draw hit markers
        for hx, hy, ht in shot_system.get_active_hit_markers(current_time):
            elapsed = current_time - ht
            alpha = 1.0 - (elapsed / HIT_MARKER_DURATION)
            draw_hit_marker(frame, hx, hy, alpha)
        
        # Draw crosshair
        draw_crosshair(frame, aim_x, aim_y, is_locked, confidence)
        
        # FPS calculation
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            current_fps = fps_counter / time_diff
            fps_counter = 0
            fps_start_time = cv2.getTickCount()
        
        # UI elements
        cv2.putText(frame, f'FPS: {int(current_fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status with confidence
        if is_locked:
            status = "TARGET LOCKED - FIRING!"
            status_color = (0, 0, 255)
        elif confidence > 0.5:
            status = f"ACQUIRING TARGET... {int(confidence * 100)}%"
            status_color = (0, 165, 255)
        elif len(faces) > 0:
            status = "TARGET DETECTED"
            status_color = (0, 255, 255)
        else:
            status = "SEARCHING..."
            status_color = (128, 128, 128)
            
        cv2.putText(frame, status, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Statistics
        cv2.putText(frame, f'Shots: {shot_system.shots_fired}', (10, FRAME_HEIGHT - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Hits: {shot_system.hits}', (10, FRAME_HEIGHT - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Accuracy: {shot_system.get_accuracy():.1f}%', (10, FRAME_HEIGHT - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display result
        cv2.imshow('AI Aiming Simulator - Enhanced', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    print("\n=== FINAL STATISTICS ===")
    print(f"Total shots: {shot_system.shots_fired}")
    print(f"Hits: {shot_system.hits}")
    print(f"Accuracy: {shot_system.get_accuracy():.1f}%")
    
    detection_thread.stop()
    detection_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
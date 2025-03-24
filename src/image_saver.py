import os
import cv2
from datetime import datetime, timedelta
import traceback
import time
from collections import defaultdict

class ImageSaver:
    def __init__(self, base_dir="Login"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.presence_times = defaultdict(float)  # Track presence duration per person
        self.last_capture_times = {}  # Track last capture time per person
        self.required_presence = 4  # Seconds of continuous presence required
        self.cooldown_period = 300  # 5 minutes cooldown (in seconds)
        self.last_seen = {}  # Track last seen time per person

    def get_today_folder(self):
        """Create and return today's date folder path"""
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_dir, today)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def update_presence(self, name, current_time):
        """Update presence tracking for a person"""
        if name not in self.last_seen:
            # New person detected
            self.presence_times[name] = 0
            self.last_seen[name] = current_time
            return False
        
        time_since_last_seen = current_time - self.last_seen[name]
        
        if time_since_last_seen < 1.0:  # Considered continuous if seen within 1 second
            self.presence_times[name] += time_since_last_seen
            self.last_seen[name] = current_time
        else:
            # Reset if presence was interrupted
            self.presence_times[name] = 0
            self.last_seen[name] = current_time
            
        return self.presence_times[name] >= self.required_presence

    def should_save_image(self, name, current_time):
        """Check if we should save image for this person"""
        # Check cooldown period first
        if name in self.last_capture_times:
            time_since_last_capture = current_time - self.last_capture_times[name]
            if time_since_last_capture < self.cooldown_period:
                return False
        
        # Then check presence duration
        return self.update_presence(name, current_time)

    def save_face_image(self, frame, face_box, name):
        """Save detected face image with name and timestamp if conditions are met"""
        try:
            current_time = time.time()
            
            if not self.should_save_image(name, current_time):
                return None

            # Prepare save path
            save_dir = self.get_today_folder()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)
            
            # Crop and save face
            x1, y1, x2, y2 = face_box
            face_img = frame[y1:y2, x1:x2]
            
            # Ensure minimum face size
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                return None
                
            success = cv2.imwrite(save_path, face_img)
            
            if success:
                self.last_capture_times[name] = current_time
                self.presence_times[name] = 0  # Reset presence timer after save
                print(f"Saved face image to: {save_path}")
                return save_path
            return None
            
        except Exception as e:
            print(f"Error saving face image: {str(e)}")
            print(traceback.format_exc())
            return None

    def get_last_capture_time(self, name=None):
        """Get formatted last capture time"""
        if name and name in self.last_capture_times:
            return datetime.fromtimestamp(self.last_capture_times[name]).strftime('%H:%M:%S')
        return "Never"
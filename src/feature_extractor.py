"""
ASL Expert - Feature Extractor
===============================
Extracts 121 geometric features from hand landmarks.

Author: ASL Expert Team
"""

import numpy as np
from typing import List


class FeatureExtractor:
    """
    Production-grade landmark feature extraction.
    
    Features extracted (calculated dynamically):
    - Normalized XY coordinates (42)
    - Fingertip-to-palm distances (5)
    - Fingertip-to-wrist distances (5)
    - Inter-fingertip distances (10)
    - Finger extension ratios (5)
    - Finger curl angles (10)
    - Thumb-finger distances (4)
    - Finger heights (5)
    - Spread angles (4)
    - Palm orientation (2)
    - Aspect ratio (1)
    - Z-depth (21)
    - Thumb direction (2)
    - Index-middle separation (1)
    - Ring-pinky separation (1)
    - Knuckle alignment (3)
    Total: 121 features
    """
    
    WRIST = 0
    THUMB = [1, 2, 3, 4]
    INDEX = [5, 6, 7, 8]
    MIDDLE = [9, 10, 11, 12]
    RING = [13, 14, 15, 16]
    PINKY = [17, 18, 19, 20]
    
    FINGERTIPS = [4, 8, 12, 16, 20]
    FINGER_MCPS = [2, 5, 9, 13, 17]
    PALM_CENTER = [0, 5, 9, 13, 17]
    
    def __init__(self):
        # Calculate actual feature count dynamically
        self.num_features = self._calculate_feature_count()
    
    def _calculate_feature_count(self) -> int:
        """Calculate the actual number of features extracted."""
        # Count features based on the extract() method implementation
        count = 0
        count += 21 * 2  # 1. Normalized XY (42)
        count += 5  # 2. Fingertip-to-palm distances (5)
        count += 5  # 3. Fingertip-to-wrist distances (5)
        count += 10  # 4. Inter-fingertip distances (C(5,2) = 10)
        count += 5  # 5. Finger extension ratios (5)
        # 6. Finger curl angles: each finger has len(finger)-2 angles
        # THUMB=[1,2,3,4] -> 2 angles, INDEX=[5,6,7,8] -> 2, etc.
        count += sum(len(finger) - 2 for finger in [self.THUMB, self.INDEX, self.MIDDLE, self.RING, self.PINKY])  # 10
        count += 4  # 7. Thumb-finger distances (4)
        count += 5  # 8. Finger heights (5)
        count += 4  # 9. Spread angles (4)
        count += 2  # 10. Palm orientation (2)
        count += 1  # 11. Aspect ratio (1)
        count += 21  # 12. Z-depth (21)
        count += 2  # 13. Thumb direction (2)
        count += 1  # 14. Index-middle separation (1)
        count += 1  # 15. Ring-pinky separation (1)
        count += 3  # 16. Knuckle alignment (3)
        return count
        
    def extract(self, landmarks: List) -> np.ndarray:
        """Extract all features from hand landmarks."""
        if landmarks is None or len(landmarks) < 21:
            return np.zeros(self.num_features, dtype=np.float32)
        
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        features = []
        
        # Normalize coordinates
        wrist = pts[self.WRIST]
        centered = pts - wrist
        scale = np.linalg.norm(pts[9] - wrist) + 1e-8
        normalized = centered / scale
        
        # 1. Normalized XY (42)
        for i in range(21):
            features.extend([normalized[i, 0], normalized[i, 1]])
        
        # 2. Fingertip-to-palm distances (5)
        palm_center = pts[self.PALM_CENTER].mean(axis=0)
        for tip in self.FINGERTIPS:
            dist = np.linalg.norm(pts[tip] - palm_center) / scale
            features.append(dist)
        
        # 3. Fingertip-to-wrist distances (5)
        for tip in self.FINGERTIPS:
            dist = np.linalg.norm(pts[tip] - wrist) / scale
            features.append(dist)
        
        # 4. Inter-fingertip distances (10)
        for i, tip1 in enumerate(self.FINGERTIPS):
            for tip2 in self.FINGERTIPS[i+1:]:
                dist = np.linalg.norm(pts[tip1] - pts[tip2]) / scale
                features.append(dist)
        
        # 5. Finger extension ratios (5)
        fingers = [self.THUMB, self.INDEX, self.MIDDLE, self.RING, self.PINKY]
        for finger in fingers:
            straight = np.linalg.norm(pts[finger[-1]] - pts[finger[0]])
            chain = sum(np.linalg.norm(pts[finger[i+1]] - pts[finger[i]]) 
                       for i in range(len(finger)-1))
            ratio = straight / (chain + 1e-8)
            features.append(ratio)
        
        # 6. Finger curl angles (15)
        for finger in fingers:
            for j in range(len(finger) - 2):
                v1 = pts[finger[j+1]] - pts[finger[j]]
                v2 = pts[finger[j+2]] - pts[finger[j+1]]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                features.append(angle / np.pi)
        
        # 7. Thumb-finger distances (4)
        thumb_tip = pts[4]
        for tip in self.FINGERTIPS[1:]:
            dist = np.linalg.norm(thumb_tip - pts[tip]) / scale
            features.append(dist)
        
        # 8. Finger heights (5)
        for tip in self.FINGERTIPS:
            height = (pts[tip, 1] - wrist[1]) / scale
            features.append(height)
        
        # 9. Spread angles (4)
        for i in range(len(self.FINGERTIPS) - 1):
            v1 = pts[self.FINGERTIPS[i]] - palm_center
            v2 = pts[self.FINGERTIPS[i+1]] - palm_center
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            features.append(angle / np.pi)
        
        # 10. Palm orientation (2)
        palm_vec = pts[9] - wrist
        palm_vec = palm_vec / (np.linalg.norm(palm_vec) + 1e-8)
        features.extend([palm_vec[0], palm_vec[1]])
        
        # 11. Aspect ratio (1)
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        features.append(x_range / (y_range + 1e-8))
        
        # 12. Z-depth (21)
        z_centered = pts[:, 2] - pts[:, 2].mean()
        z_scale = np.abs(z_centered).max() + 1e-8
        for i in range(21):
            features.append(z_centered[i] / z_scale)
        
        # 13. Thumb direction (2)
        thumb_dir = pts[4] - pts[2]
        thumb_dir = thumb_dir / (np.linalg.norm(thumb_dir) + 1e-8)
        features.extend([thumb_dir[0], thumb_dir[1]])
        
        # 14. Index-middle separation (1)
        features.append(np.linalg.norm(pts[8] - pts[12]) / scale)
        
        # 15. Ring-pinky separation (1)
        features.append(np.linalg.norm(pts[16] - pts[20]) / scale)
        
        # 16. Knuckle alignment (3)
        mcps = pts[self.FINGER_MCPS[1:]]
        mcp_center = mcps.mean(axis=0)
        for i in range(3):
            diff = mcps[i] - mcp_center
            features.append(np.linalg.norm(diff) / scale)
        
        return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    extractor = FeatureExtractor()
    print(f"Feature count: {extractor.num_features}")

"""
Optical Flow Computation Module
Compute motion features using OpenCV Farneback Optical Flow
"""

import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpticalFlowProcessor:
    """Compute optical flow features from frame sequences"""
    
    def __init__(self, threshold=0.5):
        """
        Initialize optical flow processor
        
        Args:
            threshold (float): Motion magnitude threshold
        """
        self.threshold = threshold
    
    def compute_optical_flow(self, frames):
        """
        Compute optical flow between consecutive frames
        
        Args:
            frames (np.ndarray): Array of frames (N, H, W, 3)
            
        Returns:
            dict: Contains flow, magnitude, direction, and visualization
        """
        if frames.shape[0] < 2:
            raise ValueError("At least 2 frames required for optical flow")
        
        # Convert first frame to grayscale
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        flow_list = []
        magnitude_list = []
        direction_list = []
        flow_viz_list = []
        
        for i in range(1, frames.shape[0]):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow using Farneback
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0
            )
            
            # Compute magnitude and direction
            magnitude, direction = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Store results
            flow_list.append(flow)
            magnitude_list.append(magnitude)
            direction_list.append(direction)
            
            # Create visualization
            viz = self._visualize_flow(flow, frames[i])
            flow_viz_list.append(viz)
            
            prev_gray = curr_gray
        
        return {
            'flow': np.array(flow_list),  # (N-1, H, W, 2)
            'magnitude': np.array(magnitude_list),  # (N-1, H, W)
            'direction': np.array(direction_list),  # (N-1, H, W)
            'visualization': np.array(flow_viz_list)  # (N-1, H, W, 3)
        }
    
    def _visualize_flow(self, flow, frame, scale=1):
        """
        Visualize optical flow on frame
        
        Args:
            flow (np.ndarray): Optical flow (H, W, 2)
            frame (np.ndarray): RGB frame (H, W, 3)
            scale (float): Scale factor for visualization
            
        Returns:
            np.ndarray: Visualization frame
        """
        # Create HSV image for visualization
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Compute magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        
        # Convert to BGR for display
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Ensure frame is uint8 before blending
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 0.7, bgr, 0.3, 0)
        
        return result
    
    def extract_motion_features(self, optical_flow_data):
        """
        Extract motion features from optical flow
        
        Args:
            optical_flow_data (dict): Output from compute_optical_flow
            
        Returns:
            dict: Motion features
        """
        magnitude = optical_flow_data['magnitude']
        direction = optical_flow_data['direction']
        
        features = {
            'mean_magnitude': np.mean(magnitude),
            'max_magnitude': np.max(magnitude),
            'std_magnitude': np.std(magnitude),
            'mean_direction': np.mean(direction),
            'motion_count': np.sum(magnitude > self.threshold),
            'total_frames': magnitude.shape[0],
            'average_motion_per_frame': np.mean(np.sum(magnitude > self.threshold, axis=(1, 2)))
        }
        
        return features
    
    def save_optical_flow_visualization(self, optical_flow_data, output_dir):
        """
        Save optical flow visualizations
        
        Args:
            optical_flow_data (dict): Output from compute_optical_flow
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = optical_flow_data['visualization']
        
        for i, viz in enumerate(visualizations):
            output_path = os.path.join(output_dir, f"flow_{i:04d}.jpg")
            cv2.imwrite(output_path, viz)
        
        logger.info(f"Saved {len(visualizations)} optical flow visualizations to {output_dir}")
    
    def compute_motion_summary(self, optical_flow_data):
        """
        Generate motion summary statistics
        
        Args:
            optical_flow_data (dict): Output from compute_optical_flow
            
        Returns:
            str: Human-readable motion summary
        """
        features = self.extract_motion_features(optical_flow_data)
        
        summary = f"""
MOTION ANALYSIS SUMMARY
=======================
Mean Motion Magnitude: {features['mean_magnitude']:.2f}
Max Motion Magnitude: {features['max_magnitude']:.2f}
Motion Std Dev: {features['std_magnitude']:.2f}
Avg Motion per Frame: {features['average_motion_per_frame']:.2f}
Total Frames Analyzed: {features['total_frames']}
Frames with Significant Motion: {features['motion_count']}
        """
        
        return summary.strip(), features

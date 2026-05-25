# """
# Frame Extraction Module
# Extract frames from videos using OpenCV
# """

# import cv2
# import numpy as np
# import os
# from pathlib import Path
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class FrameExtractor:
#     """Extract frames from video files"""
    
#     def __init__(self, fps=10, frame_size=(224, 224)):
#         """
#         Initialize frame extractor
        
#         Args:
#             fps (int): Frames per second to extract
#             frame_size (tuple): Target frame size (height, width)
#         """
#         self.fps = fps
#         self.frame_size = frame_size
    
#     def extract_frames(self, video_path, output_dir=None):
#         """
#         Extract frames from a video file
        
#         Args:
#             video_path (str): Path to video file
#             output_dir (str): Directory to save extracted frames
            
#         Returns:
#             np.ndarray: Array of extracted frames (N, H, W, 3)
#         """
#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"Video not found: {video_path}")
        
#         cap = cv2.VideoCapture(video_path)
        
#         if not cap.isOpened():
#             raise RuntimeError(f"Failed to open video: {video_path}")
        
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         video_fps = cap.get(cv2.CAP_PROP_FPS)
        
#         # Calculate frame sampling interval
#         frame_interval = max(1, int(video_fps / self.fps))
        
#         frames = []
#         frame_count = 0
#         extracted_count = 0
        
#         # Create output directory if provided
#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)
        
#         while True:
#             ret, frame = cap.read()
            
#             if not ret:
#                 break
            
#             # Sample frames based on fps
#             if frame_count % frame_interval == 0:
#                 # Resize frame
#                 frame_resized = cv2.resize(frame, self.frame_size)
#                 frames.append(frame_resized)
                
#                 # Save frame if output directory provided
#                 if output_dir:
#                     frame_path = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
#                     cv2.imwrite(frame_path, frame_resized)
                
#                 extracted_count += 1
            
#             frame_count += 1
        
#         cap.release()
        
#         if len(frames) == 0:
#             raise RuntimeError(f"No frames extracted from video: {video_path}")
        
#         logger.info(f"Extracted {len(frames)} frames from {video_path}")
        
#         return np.array(frames)
    
#     def extract_frames_batch(self, video_paths, output_base_dir=None):
#         """
#         Extract frames from multiple videos
        
#         Args:
#             video_paths (list): List of video file paths
#             output_base_dir (str): Base directory for output frames
            
#         Returns:
#             dict: Dictionary mapping video path to extracted frames
#         """
#         results = {}
        
#         for video_path in video_paths:
#             try:
#                 output_dir = None
#                 if output_base_dir:
#                     video_name = Path(video_path).stem
#                     output_dir = os.path.join(output_base_dir, video_name)
                
#                 frames = self.extract_frames(video_path, output_dir)
#                 results[video_path] = frames
                
#             except Exception as e:
#                 logger.error(f"Error extracting frames from {video_path}: {str(e)}")
#                 results[video_path] = None
        
#         return results


# def normalize_frame(frame):
#     """
#     Normalize frame to [0, 1] range
    
#     Args:
#         frame (np.ndarray): Input frame (H, W, 3) with values in [0, 255]
        
#     Returns:
#         np.ndarray: Normalized frame in [0, 1]
#     """
#     return frame.astype(np.float32) / 255.0


# def denormalize_frame(frame):
#     """
#     Denormalize frame from [0, 1] to [0, 255]
    
#     Args:
#         frame (np.ndarray): Normalized frame in [0, 1]
        
#     Returns:
#         np.ndarray: Frame with values in [0, 255]
#     """
#     return np.clip(frame * 255, 0, 255).astype(np.uint8)



"""
Frame Extraction Module
Extract frames from videos using OpenCV
"""

import cv2
import numpy as np
import os
from pathlib import Path
import logging

# ============================================
# LOGGING
# ============================================

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

# Disable OpenCV logs
cv2.setNumThreads(0)


class FrameExtractor:
    """Extract frames from video files"""

    def __init__(self, fps=10, frame_size=(224, 224)):

        self.fps = fps
        self.frame_size = frame_size

    def extract_frames(self, video_path, output_dir=None):

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open video: {video_path}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        video_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if video_fps <= 0:
            video_fps = 25

        frame_interval = max(
            1,
            int(video_fps / self.fps)
        )

        frames = []

        frame_count = 0
        extracted_count = 0

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:

                frame_resized = cv2.resize(
                    frame,
                    self.frame_size
                )

                frames.append(frame_resized)

                if output_dir:

                    frame_path = os.path.join(
                        output_dir,
                        f"frame_{extracted_count:04d}.jpg"
                    )

                    cv2.imwrite(
                        frame_path,
                        frame_resized
                    )

                extracted_count += 1

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(
                f"No frames extracted from video: {video_path}"
            )

        return np.array(frames)

    def extract_frames_batch(
        self,
        video_paths,
        output_base_dir=None
    ):

        results = {}

        for video_path in video_paths:

            try:

                output_dir = None

                if output_base_dir:

                    video_name = Path(video_path).stem

                    output_dir = os.path.join(
                        output_base_dir,
                        video_name
                    )

                frames = self.extract_frames(
                    video_path,
                    output_dir
                )

                results[video_path] = frames

            except Exception as e:

                results[video_path] = None

        return results


def normalize_frame(frame):

    return frame.astype(np.float32) / 255.0


def denormalize_frame(frame):

    return np.clip(
        frame * 255,
        0,
        255
    ).astype(np.uint8)
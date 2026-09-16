import cv2
import numpy as np
import time

class VideoSynchronizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.segments = {} # {step_idx: (start_time_ms, end_time_ms)}
        self.step_configs = {} # {step_idx: target_seq_len}
        self.video_path = None
        self.fps = 30.0
        self.last_step_idx = -1
        
    def analyze_video(self, video_path, progress_callback=None):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Reset analyzer for scanning
        self.analyzer.reset_state()
        
        step_detections = {i: [] for i in range(6)}
        frame_idx = 0
        scan_stride = 5 
        
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            if frame_idx % scan_stride == 0:
                frame_small = cv2.resize(frame, (320, 240))
                results = self.analyzer.process_frame(frame_small, frame_idx, effective_fps=self.fps)
                pred_step = results["predicted_step"]
                conf = results["confidence"]
                
                if pred_step != -1 and conf > 0.5:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                    step_detections[pred_step].append(timestamp)
            
            frame_idx += 1
            if progress_callback and frame_idx % 100 == 0:
                # Minimal feedback
                pass
                
        cap.release()
        self.analyzer.reset_state()
        
        last_end_time = 0
        for i in range(6):
            times = step_detections[i]
            if not times:
                start = last_end_time
                end = last_end_time + 5000 # Default 5s if missing
            else:
                start = times[0]
                end = times[-1]
                if start < last_end_time: start = last_end_time
                if end < start: end = start + 1000
            
            self.segments[i] = (start, end)
            
            # Calculate optimal seq_len_req
            # We want Protocol to finish exactly at end of segment.
            # Duration in seconds
            dur_sec = (end - start) / 1000.0
            # Total frames in this segment
            total_seg_frames = int(dur_sec * self.fps)
            
            # We set seq_len_req to be slightly less than total frames to ensure completion
            # e.g. 90% of frames
            target_seq = max(10, int(total_seg_frames * 0.9))
            self.step_configs[i] = target_seq
            
            last_end_time = end
            
        print("Segments:", self.segments)
        print("Configs:", self.step_configs)
        return True

    def get_current_step_from_time(self, current_time_ms):
        """
        Determines the expected step based on video timestamp.
        """
        for i in range(6):
            if i in self.segments:
                start, end = self.segments[i]
                if start <= current_time_ms <= end:
                    return i
        return -1

    def sync_step(self, video_time_ms, protocol_step, protocol_progress):
        """
        Main synchronization logic.
        """
        # 1. Update Analyzer Configuration if step changed
        if protocol_step != self.last_step_idx:
            if protocol_step in self.step_configs:
                target_seq = self.step_configs[protocol_step]
                # Update the analyzer's requirement
                self.analyzer.seq_len_req = target_seq
                # Also scale YOLO requirement?
                self.analyzer.yolo_peaks_req = max(3, int(target_seq / 10))
            self.last_step_idx = protocol_step

        target_step = self.get_current_step_from_time(video_time_ms)
        
        # If we are in a gap between steps or outside known segments
        if target_step == -1:
             # Just play, maybe we are transitioning
             return 'play', 1
             
        # If Protocol is behind Video Target
        if protocol_step < target_step:
            # FORCE CATCH UP
            return 'force_next_step', 1
            
        # If Protocol is ahead of Video Target
        if protocol_step > target_step:
            # Video is behind Protocol. We MUST keep playing video to catch up.
            # Do NOT pause video.
            return 'play', 1 
            
        # Same step: Check progress
        start_ms, end_ms = self.segments[target_step]
        video_progress = (video_time_ms - start_ms) / (end_ms - start_ms + 1e-6)
        video_progress = max(0.0, min(1.0, video_progress))
        
        diff = video_progress - protocol_progress
        
        if diff > 0.2: # Video ahead
            return 'slow_video', 1
        elif diff < -0.2: # Video behind
            return 'fast_video', 1
            
        return 'play', 1

# heron_model_with_trajectory.py
"""
Heron VLM Model Manager with Trajectory Visualization
Supports trajectory drawing on 4 frames for auto-annotation
"""

import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from datetime import datetime

# Import trajectory visualization
from visual_prompting import TrajectoryVisualizer

# Heronモデルはtransformersから直接ロード可能
HERON_AVAILABLE = True
try:
    import torch
    import transformers
    logging.info("torch and transformers available - Heron can be loaded")
except ImportError as e:
    HERON_AVAILABLE = False
    logging.warning(f"Required libraries not found: {e}. Auto-annotation will not be available.")

from config import (
    HERON_MODEL_ID,
    USE_GPU,
    TORCH_DTYPE,
    NUM_FRAMES_TO_EXTRACT,
    NUM_FRAMES_TO_USE,
    USE_MULTI_FRAME,
    PROMPT_STAGE1_TEMPLATE,
    PROMPT_STAGE2_ROTATION_TEMPLATE,
    PROMPT_STAGE2_NONROTATION_TEMPLATE,
    ENABLE_FLASH_ATTENTION,
    USE_L2M_COT,
    USE_VLM_DIRECT,
    USE_SENSOR_ONLY_BASELINE,
    MAX_NEW_TOKENS_STANDARD,
    PROMPT_VERSION,
)

logger = logging.getLogger(__name__)

MACRO_GROUP_LABELS = {
    "A": "直線系",
    "B": "左回転系",
    "C": "右回転系",
    "D": "その他",
}

MACRO_GROUP_TO_CANONICAL_LABEL = {
    "A": 1,  # 直線系 representative
    "B": 6,  # 左回転系 representative
    "C": 7,  # 右回転系 representative
    "D": 4,  # その他 representative
}

FINE_LABEL_TO_MACRO_GROUP = {
    1: "A", 2: "A", 3: "A",
    6: "B", 8: "B",
    7: "C", 9: "C", 10: "C",
    0: "D", 4: "D", 5: "D",
}

STAGE1_LABELS = {
    "S": "直線系",
    "R": "回転系",
    "O": "その他",
}


class HeronAnnotatorWithTrajectory:
    """
    Heron VLM-based automatic annotation system with trajectory visualization
    Supports multi-frame input with trajectory drawing
    """
    
    def __init__(self, save_trajectory_frames: bool = True, trajectory_output_dir: str = "trajectory_frames"):
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._device = None
        self._is_heron = False
        self._last_prediction_details: Optional[Dict[str, Any]] = None
        
        # Trajectory visualization settings
        self.save_trajectory_frames = save_trajectory_frames
        self.trajectory_output_dir = trajectory_output_dir
        self.trajectory_visualizer = None
        
        # Create output directory for trajectory frames
        if self.save_trajectory_frames and not os.path.exists(self.trajectory_output_dir):
            os.makedirs(self.trajectory_output_dir)
            logger.info(f"Created trajectory frames directory: {self.trajectory_output_dir}")
        
        # Initialize trajectory visualizer
        self._init_trajectory_visualizer()
    
    def _init_trajectory_visualizer(self):
        """Initialize trajectory visualization with camera parameters"""
        # Camera calibration parameters (from batch_visualize_trajectories.py)
        fx = fy = 600.0
        width, height = 1920, 1080
        cx = width / 2.0  # 960
        cy = height * 0.75  # 810 (75% down for road focus)
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((1, 5), dtype=np.float32)
        
        # Coordinate transformation: vehicle -> camera
        # Vehicle: X=forward, Y=left, Z=up
        # Camera: X=right, Y=down, Z=forward
        
        # Camera pitch: 15 degrees downward from horizontal
        pitch_angle = np.deg2rad(15)  # 下向き（正の角度）
        cos_p = np.cos(pitch_angle)
        sin_p = np.sin(pitch_angle)
        
        # 変換行列を正しい順序で構築
        # 1. 車両座標系からカメラ座標系への基本変換（軸の再配置）
        # 2. ピッチ回転を適用
        R = np.array([
            [0, -1, 0],           # Camera X = -Vehicle Y (右方向)
            [-sin_p, 0, -cos_p],  # Camera Y = 下向き成分
            [cos_p, 0, -sin_p]    # Camera Z = 前方成分（投影方向）
        ], dtype=np.float32)
        
        # Camera position in vehicle frame (meters)
        camera_position = np.array([1.0, 0.0, 1.4], dtype=np.float32)
        t = -R @ camera_position
        
        T_v2c = np.eye(4, dtype=np.float32)
        T_v2c[:3, :3] = R
        T_v2c[:3, 3] = t
        
        self.trajectory_visualizer = TrajectoryVisualizer(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            T_v2c=T_v2c,
            image_size=(width, height)
        )
        logger.info("Trajectory visualizer initialized")
    
    @property
    def model(self):
        return self._model
    
    @property
    def processor(self):
        return self._processor
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def device(self):
        return self._device
    
    @property
    def is_heron(self):
        return self._is_heron
    
    @property
    def is_loaded(self):
        return self._model is not None and self._processor is not None

    @property
    def last_prediction_details(self) -> Optional[Dict[str, Any]]:
        return self._last_prediction_details
    
    def load_model(self):
        """Load Heron VLM model and processor"""
        if not HERON_AVAILABLE:
            raise RuntimeError("Required libraries (torch, transformers) not available")
        
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading model: {HERON_MODEL_ID}")
            
            # Device setup
            if USE_GPU and torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self._device = torch.device("cpu")
                logger.info("Using CPU")
            
            # Check if Qwen2-VL model
            is_qwen = "qwen" in HERON_MODEL_ID.lower()
            
            if is_qwen:
                # Qwen2-VL specific loading
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    HERON_MODEL_ID,
                    torch_dtype=TORCH_DTYPE,
                    device_map="auto" if USE_GPU else None
                )
                self._processor = AutoProcessor.from_pretrained(HERON_MODEL_ID)
                logger.info(f"Qwen2-VL model loaded with multi-frame support (memory optimized)")
            else:
                # Standard Heron model loading
                self._model = AutoModelForCausalLM.from_pretrained(
                    HERON_MODEL_ID,
                    torch_dtype=TORCH_DTYPE,
                    trust_remote_code=True,
                    device_map="auto" if USE_GPU else None
                )
                self._processor = AutoProcessor.from_pretrained(
                    HERON_MODEL_ID,
                    trust_remote_code=True
                )
                self._is_heron = True
                logger.info(f"Heron model loaded with multi-frame support")
            
            if not USE_GPU or self._device.type == "cpu":
                self._model = self._model.to(self._device)
            
            self._model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_frames_from_video(
        self, 
        video_path: str, 
        num_frames: int = NUM_FRAMES_TO_USE,
        start_time: float = 0.0,
        duration: float = 5.0
    ) -> Tuple[List[Image.Image], List[int]]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: 4)
            start_time: Start time in seconds
            duration: Duration to sample from in seconds
        
        Returns:
            Tuple of (List of PIL Image objects, List of frame indices)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame indices
            start_frame = int(start_time * fps)
            available_duration = min(duration, video_duration - start_time)
            end_frame = min(int((start_time + available_duration) * fps), total_frames)
            
            if end_frame <= start_frame:
                end_frame = min(start_frame + 2, total_frames)
            
            frame_indices = np.linspace(
                start_frame, 
                end_frame - 1, 
                num_frames, 
                dtype=int
            )
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                else:
                    logger.warning(f"Failed to read frame {frame_idx}")
            
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames, frame_indices.tolist()
            
        finally:
            cap.release()
    
    def draw_trajectory_on_frames(
        self,
        frames: List[Image.Image],
        sensor_data: Dict[str, Any],
        sample_id: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Draw trajectory on all frames
        
        Args:
            frames: List of PIL Images
            sensor_data: Dictionary containing sensor data (speed, acc_x, gyro_z, etc.)
            sample_id: Optional sample ID for saving trajectory frames
        
        Returns:
            List of PIL Images with trajectory drawn
        """
        if not frames:
            logger.warning("No frames to draw trajectory on")
            return frames
        
        # Extract sensor data for trajectory calculation
        speed = sensor_data.get('speed', 0.0)  # km/h
        gyro_z = sensor_data.get('gyro_z', 0.0)  # rad/s (yaw rate)
        
        # Convert speed to m/s
        speed_ms = speed / 3.6
        
        # Create speed and yaw rate sequences for 3 seconds (30 steps * 0.1s)
        num_steps = 30
        speed_seq = np.full(num_steps, speed_ms, dtype=np.float32)
        yaw_rate_seq = np.full(num_steps, gyro_z, dtype=np.float32)
        
        # Calculate 3D trajectory using Bicycle Model
        trajectory_3d = self.trajectory_visualizer.calculate_trajectory(speed_seq, yaw_rate_seq)
        
        # Project to 2D image coordinates
        image_points, valid_mask = self.trajectory_visualizer.project_3d_to_2d(trajectory_3d)
        
        visible_count = np.sum(valid_mask)
        logger.info(f"Trajectory: {visible_count}/{len(valid_mask)} points visible")

        self._last_prediction_details = {
            **(self._last_prediction_details or {}),
            "visible_trajectory_points": int(visible_count),
            "trajectory_total_points": int(len(valid_mask)),
        }
        
        # Draw trajectory on each frame
        frames_with_trajectory = []
        for frame_idx, frame in enumerate(frames):
            # Convert PIL to numpy array for drawing
            frame_np = np.array(frame)
            
            # Draw trajectory using the updated method (points + lines)
            result_frame = self.trajectory_visualizer.draw_trajectory_on_image(
                image=frame_np,
                trajectory_3d=trajectory_3d,
                color=(255, 0, 0),  # Red trajectory (RGB)
                thickness=10,
                point_radius=10
            )

            self._draw_motion_cue_panel(
                result_frame,
                speed=speed,
                gyro_z=gyro_z,
                acc_x=float(sensor_data.get('acc_x', 0.0)),
                latitude=float(sensor_data.get('latitude', 0.0)),
                longitude=float(sensor_data.get('longitude', 0.0)),
            )
            
            # Keep the on-image text short so the trajectory itself stays salient.
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (12, 10), (260, 42), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.42, result_frame, 0.58, 0, result_frame)
            cv2.putText(
                result_frame,
                f"frame {frame_idx+1}/4",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            # Convert back to PIL
            result_pil = Image.fromarray(result_frame)
            frames_with_trajectory.append(result_pil)
            
            # Save trajectory frame if requested
            if self.save_trajectory_frames and sample_id is not None:
                # Ensure output directory exists
                os.makedirs(self.trajectory_output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"sample_{sample_id:03d}_frame_{frame_idx+1}_traj_{timestamp}.jpg"
                output_path = os.path.join(self.trajectory_output_dir, output_filename)
                result_pil.save(output_path, quality=95)
                logger.debug(f"Saved trajectory frame: {output_path}")
        
        logger.info(f"Drew trajectory on {len(frames_with_trajectory)} frames")
        return frames_with_trajectory

    def prepare_visual_inputs(
        self,
        frames: List[Image.Image],
        sensor_data: Dict[str, Any],
        sample_id: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Prepare model inputs as:
        - 4 original frames
        - 1 trajectory-only summary image
        """
        if not frames:
            return frames

        speed = sensor_data.get('speed', 0.0)
        gyro_z = sensor_data.get('gyro_z', 0.0)
        speed_ms = speed / 3.6

        num_steps = 30
        speed_seq = np.full(num_steps, speed_ms, dtype=np.float32)
        yaw_rate_seq = np.full(num_steps, gyro_z, dtype=np.float32)
        trajectory_3d = self.trajectory_visualizer.calculate_trajectory(speed_seq, yaw_rate_seq)
        _, valid_mask = self.trajectory_visualizer.project_3d_to_2d(trajectory_3d)
        visible_count = int(np.sum(valid_mask))
        trajectory_features = self._compute_trajectory_features(trajectory_3d)
        logger.info(f"Trajectory summary: {visible_count}/{len(valid_mask)} points visible")

        summary_np = self.trajectory_visualizer.render_trajectory_summary(
            trajectory_3d=trajectory_3d,
            speed=float(speed),
            acc_x=float(sensor_data.get('acc_x', 0.0)),
            gyro_z=float(gyro_z),
            latitude=float(sensor_data.get('latitude', 0.0)),
            longitude=float(sensor_data.get('longitude', 0.0)),
        )
        summary_pil = Image.fromarray(summary_np)

        self._last_prediction_details = {
            **(self._last_prediction_details or {}),
            "visible_trajectory_points": visible_count,
            "trajectory_total_points": int(len(valid_mask)),
            "trajectory_features": trajectory_features,
            "visual_input_mode": "raw_frames_plus_summary",
            "num_visual_inputs": len(frames) + 1,
        }

        if self.save_trajectory_frames and sample_id is not None:
            os.makedirs(self.trajectory_output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sample_{sample_id:03d}_trajectory_summary_{timestamp}.jpg"
            output_path = os.path.join(self.trajectory_output_dir, output_filename)
            summary_pil.save(output_path, quality=95)
            logger.debug(f"Saved trajectory summary image: {output_path}")

        return [frame.copy() for frame in frames] + [summary_pil]

    def _compute_trajectory_features(self, trajectory_3d: np.ndarray) -> Dict[str, float]:
        """Compute deterministic geometry features from the predicted trajectory."""
        points_xy = np.asarray(trajectory_3d[:, :2], dtype=np.float32)
        if len(points_xy) < 2:
            return {
                "forward_distance_m": 0.0,
                "end_lateral_offset_m": 0.0,
                "heading_delta_rad": 0.0,
                "path_length_m": 0.0,
                "curvature_score": 0.0,
            }

        deltas = np.diff(points_xy, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        path_length = float(np.sum(segment_lengths))
        end_x = float(points_xy[-1, 0] - points_xy[0, 0])
        end_y = float(points_xy[-1, 1] - points_xy[0, 1])

        headings = np.unwrap(np.arctan2(deltas[:, 1], np.maximum(deltas[:, 0], 1e-6)))
        heading_delta = float(headings[-1] - headings[0]) if len(headings) > 0 else 0.0
        curvature_score = abs(heading_delta) / max(path_length, 1e-6)

        return {
            "forward_distance_m": end_x,
            "end_lateral_offset_m": end_y,
            "heading_delta_rad": heading_delta,
            "path_length_m": path_length,
            "curvature_score": float(curvature_score),
        }

    def _determine_stage1_group(
        self,
        speed: float,
        gyro_z: float,
        visible_points: int,
        trajectory_features: Dict[str, float],
    ) -> Tuple[str, Dict[str, Any]]:
        """Deterministically classify stage1 into S / R / O."""
        forward_distance = float(trajectory_features.get("forward_distance_m", 0.0))
        lateral_offset = abs(float(trajectory_features.get("end_lateral_offset_m", 0.0)))
        heading_delta = abs(float(trajectory_features.get("heading_delta_rad", 0.0)))
        curvature_score = float(trajectory_features.get("curvature_score", 0.0))
        abs_gyro = abs(float(gyro_z))
        lateral_ratio = lateral_offset / max(forward_distance, 1.0)

        reason = {
            "speed": float(speed),
            "visible_points": int(visible_points),
            "forward_distance_m": forward_distance,
            "abs_lateral_offset_m": lateral_offset,
            "lateral_ratio": lateral_ratio,
            "abs_heading_delta_rad": heading_delta,
            "abs_gyro_z": abs_gyro,
            "curvature_score": curvature_score,
            "turn_votes": [],
        }

        if speed < 1.0 or (speed < 1.5 and visible_points == 0) or (speed < 2.0 and forward_distance < 1.0):
            reason["decision"] = "O"
            reason["rule"] = "low_speed_or_near_zero_motion"
            return "O", reason

        if lateral_offset >= 1.2 and lateral_ratio >= 0.18:
            reason["turn_votes"].append("lateral_ratio")
        if heading_delta >= 0.20:
            reason["turn_votes"].append("heading_delta")
        if abs_gyro >= 0.18 and forward_distance >= 2.0:
            reason["turn_votes"].append("gyro")
        if curvature_score >= 0.04 and forward_distance >= 2.0:
            reason["turn_votes"].append("curvature")

        has_anchor = any(v in reason["turn_votes"] for v in ("lateral_ratio", "gyro"))
        strong_turn = (
            (len(reason["turn_votes"]) >= 2 and has_anchor)
            or (lateral_ratio >= 0.28 and heading_delta >= 0.14)
            or (heading_delta >= 0.24 and abs_gyro >= 0.16)
            or (lateral_ratio >= 0.20 and curvature_score >= 0.04)
        )

        if strong_turn:
            reason["decision"] = "R"
            reason["rule"] = "turn_geometry_detected"
            return "R", reason

        reason["decision"] = "S"
        reason["rule"] = "default_straight_motion"
        return "S", reason

    def _determine_fine_label_from_macro(
        self,
        macro_group: str,
        speed: float,
        acc_x: float,
        gyro_z: float,
        visible_points: int,
        trajectory_features: Dict[str, float],
    ) -> Tuple[int, Dict[str, Any]]:
        """Map macro group to a finer 11-class label deterministically."""
        forward_distance = float(trajectory_features.get("forward_distance_m", 0.0))
        lateral = float(trajectory_features.get("end_lateral_offset_m", 0.0))
        lateral_ratio = abs(lateral) / max(forward_distance, 1.0)
        heading = float(trajectory_features.get("heading_delta_rad", 0.0))
        heading_abs = abs(heading)
        curvature = float(trajectory_features.get("curvature_score", 0.0))
        abs_gyro = abs(float(gyro_z))

        reason = {
            "macro_group": macro_group,
            "speed": float(speed),
            "acc_x": float(acc_x),
            "gyro_z": float(gyro_z),
            "visible_points": int(visible_points),
            "forward_distance_m": forward_distance,
            "end_lateral_offset_m": lateral,
            "lateral_ratio": lateral_ratio,
            "heading_delta_rad": heading,
            "curvature_score": curvature,
        }

        if macro_group == "A":
            if acc_x <= -0.18 and speed >= 4.0:
                reason["rule"] = "straight_deceleration"
                return 3, reason
            if acc_x >= 0.12 and speed <= 35.0:
                reason["rule"] = "straight_acceleration"
                return 2, reason
            reason["rule"] = "straight_constant"
            return 1, reason

        if macro_group == "D":
            if speed < 1.0 or visible_points == 0 or forward_distance < 0.8:
                reason["rule"] = "other_stop"
                return 4, reason
            if speed < 5.0 and acc_x > 0.12 and visible_points > 0:
                reason["rule"] = "other_start"
                return 5, reason
            reason["rule"] = "other_misc"
            return 0, reason

        if macro_group == "B":
            if heading_abs >= 0.35 or curvature >= 0.06 or abs_gyro >= 0.20:
                reason["rule"] = "left_turn"
                return 6, reason
            if lateral_ratio >= 0.12:
                reason["rule"] = "left_lane_change"
                return 8, reason
            reason["rule"] = "left_turn_fallback"
            return 6, reason

        if macro_group == "C":
            if heading_abs >= 1.20 and forward_distance < 8.0 and lateral_ratio >= 0.20:
                reason["rule"] = "u_turn"
                return 10, reason
            if heading_abs >= 0.35 or curvature >= 0.06 or abs_gyro >= 0.20:
                reason["rule"] = "right_turn"
                return 7, reason
            if lateral_ratio >= 0.12:
                reason["rule"] = "right_lane_change"
                return 9, reason
            reason["rule"] = "right_turn_fallback"
            return 7, reason

        reason["rule"] = "default_misc"
        return 0, reason

    def _classify_sensor_only_label(
        self,
        sensor_data: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """Classify a single 11-class label from sensor values only."""
        speed = float(sensor_data.get("speed", 0.0))
        acc_x = float(sensor_data.get("acc_x", 0.0))
        gyro_z = float(sensor_data.get("gyro_z", 0.0))
        abs_gyro = abs(gyro_z)

        reason = {
            "speed_kmh": speed,
            "acc_x": acc_x,
            "gyro_z": gyro_z,
            "abs_gyro_z": abs_gyro,
        }

        # Very low speed: stop/start/misc around standstill
        if speed <= 1.5:
            if acc_x >= 0.18:
                reason["rule"] = "standstill_start"
                return 5, reason
            reason["rule"] = "standstill_stop"
            return 4, reason

        # Strong yaw at modest speed can indicate a U-turn-like maneuver.
        if abs_gyro >= 0.45 and speed <= 18.0:
            reason["rule"] = "strong_yaw_uturn"
            return 10, reason

        # Turn / lane change decisions from yaw rate only.
        if abs_gyro >= 0.16:
            reason["rule"] = "strong_yaw_turn"
            return (6 if gyro_z > 0 else 7), reason

        if abs_gyro >= 0.07 and speed >= 15.0:
            reason["rule"] = "moderate_yaw_lane_change"
            return (8 if gyro_z > 0 else 9), reason

        # Straight speed-event classes.
        if acc_x >= 0.25:
            reason["rule"] = "straight_acceleration"
            return 2, reason

        if acc_x <= -0.35:
            if speed <= 6.0:
                reason["rule"] = "slow_decel_stop_like"
                return 4, reason
            reason["rule"] = "straight_deceleration"
            return 3, reason

        # Slight decel near zero motion often behaves like stop-like.
        if speed <= 4.0 and acc_x < -0.12:
            reason["rule"] = "low_speed_stop_like"
            return 4, reason

        if abs(acc_x) <= 0.08 and abs_gyro <= 0.03:
            reason["rule"] = "steady_constant_speed"
            return 1, reason

        if acc_x < 0:
            reason["rule"] = "fallback_deceleration"
            return 3, reason

        if acc_x > 0:
            reason["rule"] = "fallback_acceleration"
            return 2, reason

        reason["rule"] = "fallback_constant_speed"
        return 1, reason

    def _predict_sensor_only_baseline(
        self,
        sensor_data: Dict[str, Any],
        sample_id: Optional[int] = None,
    ) -> Optional[int]:
        """Predict a single 11-class label from sensor values only."""
        action_label, sensor_rule_reason = self._classify_sensor_only_label(sensor_data)
        self._last_prediction_details = {
            **(self._last_prediction_details or {}),
            "prompt_version": PROMPT_VERSION,
            "visual_input_mode": "sensor_only",
            "predicted_label": action_label,
            "extracted_label": action_label,
            "generated_text": None,
            "prompt_text": None,
            "macro_prediction_code": FINE_LABEL_TO_MACRO_GROUP.get(action_label),
            "macro_prediction_label": MACRO_GROUP_LABELS.get(FINE_LABEL_TO_MACRO_GROUP.get(action_label)),
            "sensor_rule_reason": sensor_rule_reason,
            "error": None,
        }
        logger.info(f"[Sensor-only baseline] Predicted action label: {action_label}")
        return action_label

    def _determine_rotation_direction(
        self,
        gyro_z: float,
        trajectory_features: Dict[str, float],
    ) -> Tuple[str, Dict[str, Any]]:
        """Deterministically classify left/right direction from signed geometry."""
        lateral = float(trajectory_features.get("end_lateral_offset_m", 0.0))
        heading = float(trajectory_features.get("heading_delta_rad", 0.0))
        forward_distance = float(trajectory_features.get("forward_distance_m", 0.0))
        lateral_ratio = abs(lateral) / max(forward_distance, 1.0)

        left_score = 0.0
        right_score = 0.0
        evidence: List[str] = []

        if abs(lateral) >= 1.2 and lateral_ratio >= 0.12:
            if lateral > 0:
                left_score += 2.0
                evidence.append("lateral_left_strong")
            else:
                right_score += 2.0
                evidence.append("lateral_right_strong")
        elif abs(lateral) >= 0.8 and lateral_ratio >= 0.08:
            if lateral > 0:
                left_score += 1.0
                evidence.append("lateral_left")
            else:
                right_score += 1.0
                evidence.append("lateral_right")

        if abs(heading) >= 0.18:
            if heading > 0:
                left_score += 2.0
                evidence.append("heading_left_strong")
            else:
                right_score += 2.0
                evidence.append("heading_right_strong")
        elif abs(heading) >= 0.10:
            if heading > 0:
                left_score += 1.0
                evidence.append("heading_left")
            else:
                right_score += 1.0
                evidence.append("heading_right")

        if abs(gyro_z) >= 0.18:
            if gyro_z > 0:
                left_score += 1.5
                evidence.append("gyro_left_strong")
            else:
                right_score += 1.5
                evidence.append("gyro_right_strong")
        elif abs(gyro_z) >= 0.10:
            if gyro_z > 0:
                left_score += 0.5
                evidence.append("gyro_left")
            else:
                right_score += 0.5
                evidence.append("gyro_right")

        direction = "AMB"
        confidence = "low"
        if max(left_score, right_score) >= 2.0 and abs(left_score - right_score) >= 1.0:
            direction = "L" if left_score > right_score else "R"
            confidence = "high"
        elif max(left_score, right_score) >= 1.5 and abs(left_score - right_score) >= 0.5:
            direction = "L" if left_score > right_score else "R"
            confidence = "medium"

        reason = {
            "end_lateral_offset_m": lateral,
            "heading_delta_rad": heading,
            "gyro_z": float(gyro_z),
            "forward_distance_m": forward_distance,
            "lateral_ratio": lateral_ratio,
            "left_score": left_score,
            "right_score": right_score,
            "evidence": evidence,
            "confidence": confidence,
        }
        return direction, reason

    def _classify_macro_group_deterministic(
        self,
        speed: float,
        acc_x: float,
        gyro_z: float,
        visible_points: int,
        trajectory_features: Dict[str, float],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Deterministic 4-group classifier without VLM."""
        stage_traces: List[Dict[str, Any]] = []

        stage1_choice, stage1_reason = self._determine_stage1_group(
            speed=float(speed),
            gyro_z=float(gyro_z),
            visible_points=visible_points,
            trajectory_features=trajectory_features,
        )
        stage_traces.append({
            "stage": "stage1_deterministic",
            "output": stage1_choice,
            "reason": stage1_reason,
        })

        if stage1_choice == "O":
            stage_traces.append({
                "stage": "stage2_other_deterministic",
                "output": "D",
                "reason": {
                    "rule": "stage1_other",
                    "speed": float(speed),
                    "visible_points": int(visible_points),
                },
            })
            return "D", stage_traces

        if stage1_choice == "R":
            rotation_choice, rotation_reason = self._determine_rotation_direction(
                gyro_z=float(gyro_z),
                trajectory_features=trajectory_features,
            )
            stage_traces.append({
                "stage": "stage2_rotation_deterministic",
                "output": rotation_choice,
                "reason": rotation_reason,
            })
            if rotation_choice == "L":
                return "B", stage_traces
            if rotation_choice == "R":
                return "C", stage_traces

            lateral = float(trajectory_features.get("end_lateral_offset_m", 0.0))
            fallback_group = "B" if lateral >= 0 else "C"
            stage_traces.append({
                "stage": "stage2_rotation_fallback",
                "output": fallback_group,
                "reason": {
                    "rule": "signed_lateral_offset_fallback",
                    "end_lateral_offset_m": lateral,
                    "gyro_z": float(gyro_z),
                },
            })
            return fallback_group, stage_traces

        straight_reason = {
            "rule": "stage1_straight_motion",
            "speed": float(speed),
            "acc_x": float(acc_x),
            "visible_points": int(visible_points),
            "forward_distance_m": float(trajectory_features.get("forward_distance_m", 0.0)),
        }
        stage_traces.append({
            "stage": "stage2_straight_deterministic",
            "output": "A",
            "reason": straight_reason,
        })
        return "A", stage_traces

    def _draw_motion_cue_panel(
        self,
        image: np.ndarray,
        speed: float,
        gyro_z: float,
        acc_x: float,
        latitude: float,
        longitude: float,
    ) -> None:
        """Draw a large event panel so speed and turn cues survive resizing."""
        h, w = image.shape[:2]
        panel_w = 420
        panel_h = 146
        x1 = 24
        y1 = h - panel_h - 28
        x2 = x1 + panel_w
        y2 = y1 + panel_h

        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.46, image, 0.54, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        hints = self._derive_event_hints(speed, acc_x, gyro_z)
        cv2.putText(
            image,
            f"speed {hints['speed_event_text']}",
            (x1 + 14, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.74,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            image,
            f"turn {hints['turn_event_text']}",
            (x1 + 214, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.74,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        bar_x1 = x1 + 18
        bar_y1 = y1 + 46
        bar_x2 = x2 - 18
        bar_y2 = bar_y1 + 26
        cv2.rectangle(image, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
        cv2.rectangle(image, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)
        fill_ratio = max(0.0, min(speed / 50.0, 1.0))
        fill_x2 = int(bar_x1 + (bar_x2 - bar_x1) * fill_ratio)
        if fill_x2 > bar_x1:
            cv2.rectangle(image, (bar_x1, bar_y1), (fill_x2, bar_y2), hints["speed_event_color"], -1)

        arrow_center_x = x2 - 48
        arrow_center_y = y1 + 96
        self._draw_direction_arrow(
            image,
            center=(arrow_center_x, arrow_center_y),
            direction=hints["turn_arrow_direction"],
            color=hints["turn_event_color"]
        )

        cv2.circle(image, (x1 + 34, y1 + 94), 12, (0, 200, 255), -1)
        cv2.circle(image, (x1 + 34, y1 + 94), 14, (255, 255, 255), 2)
        cv2.putText(
            image,
            f"speed {speed:.0f} km/h",
            (x1 + 58, y1 + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            image,
            f"acc {acc_x:+.2f}",
            (x1 + 58, y1 + 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            image,
            f"yaw {gyro_z:+.2f}",
            (x1 + 58, y1 + 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            image,
            f"gps {latitude:.5f}, {longitude:.5f}",
            (x1 + 180, y1 + 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def _draw_direction_arrow(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        direction: str,
        color: Tuple[int, int, int],
    ) -> None:
        cx, cy = center
        if direction == "left":
            pts = np.array([
                (cx + 22, cy - 12),
                (cx - 6, cy - 12),
                (cx - 6, cy - 22),
                (cx - 28, cy),
                (cx - 6, cy + 22),
                (cx - 6, cy + 12),
                (cx + 22, cy + 12),
            ], dtype=np.int32)
        elif direction == "right":
            pts = np.array([
                (cx - 22, cy - 12),
                (cx + 6, cy - 12),
                (cx + 6, cy - 22),
                (cx + 28, cy),
                (cx + 6, cy + 22),
                (cx + 6, cy + 12),
                (cx - 22, cy + 12),
            ], dtype=np.int32)
        else:
            pts = np.array([
                (cx - 24, cy - 10),
                (cx + 2, cy - 10),
                (cx + 2, cy - 22),
                (cx + 26, cy),
                (cx + 2, cy + 22),
                (cx + 2, cy + 10),
                (cx - 24, cy + 10),
            ], dtype=np.int32)

        cv2.fillPoly(image, [pts], color)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    def _derive_event_hints(
        self,
        speed: float,
        acc_x: float,
        gyro_z: float,
    ) -> Dict[str, Any]:
        if speed < 1.5:
            speed_event_text = "stop-like"
            speed_event_color = (60, 60, 255)
        elif (speed < 10.0 and acc_x < -0.12) or acc_x < -0.35:
            speed_event_text = "decel-like"
            speed_event_color = (0, 160, 255)
        elif (speed < 6.0 and acc_x > 0.15) or acc_x > 0.30:
            speed_event_text = "accel-like"
            speed_event_color = (0, 210, 120)
        else:
            speed_event_text = "cruise-like"
            speed_event_color = (0, 200, 255)

        gyro_left = gyro_z > 0.14
        gyro_right = gyro_z < -0.14

        turn_arrow_direction = "straight"
        turn_event_text = "straight-like"
        turn_event_color = (180, 180, 180)
        if gyro_left:
            turn_arrow_direction = "left"
            turn_event_text = "left-like"
            turn_event_color = (255, 180, 0)
        elif gyro_right:
            turn_arrow_direction = "right"
            turn_event_text = "right-like"
            turn_event_color = (255, 180, 0)

        return {
            "speed_event_text": speed_event_text,
            "speed_event_color": speed_event_color,
            "turn_event_text": turn_event_text,
            "turn_event_color": turn_event_color,
            "turn_arrow_direction": turn_arrow_direction,
        }

    def _build_stage1_candidates(
        self,
        hints: Dict[str, Any],
        visible_trajectory_points: int,
    ) -> List[str]:
        """Stage 1 decides between straight / rotation / other."""
        speed_hint = hints["speed_event_text"]
        turn_hint = hints["turn_event_text"]

        if visible_trajectory_points == 0 and speed_hint in {"stop-like", "decel-like"}:
            return ["O", "S"]
        if speed_hint == "stop-like":
            return ["O", "S"]
        if turn_hint in {"left-like", "right-like", "ambiguous-like"}:
            return ["R", "S"]
        if speed_hint in {"decel-like", "accel-like"}:
            return ["S", "O"]
        return ["S", "O"]

    def _format_stage1_candidate_lines(self, stage1_candidates: List[str]) -> str:
        descriptions = {
            "S": "S: 直線系（等速・加速・減速）",
            "R": "R: 回転系（左折・右折・車線変更・転回）",
            "O": "O: その他（停止・発進・その他）",
        }
        return "\n".join(descriptions[code] for code in stage1_candidates)

    def _build_rotation_candidates(self, hints: Dict[str, Any]) -> List[str]:
        turn_hint = hints["turn_event_text"]
        if turn_hint == "left-like":
            return ["L", "R"]
        if turn_hint == "right-like":
            return ["R", "L"]
        return ["L", "R"]

    def _format_rotation_candidate_lines(self, rotation_candidates: List[str]) -> str:
        descriptions = {
            "L": "L: 左回転系（左折・左車線変更）",
            "R": "R: 右回転系（右折・右車線変更・転回）",
        }
        return "\n".join(descriptions[code] for code in rotation_candidates)

    def _build_nonrotation_candidates(
        self,
        hints: Dict[str, Any],
        visible_trajectory_points: int,
    ) -> List[str]:
        speed_hint = hints["speed_event_text"]
        if visible_trajectory_points == 0 and speed_hint in {"stop-like", "decel-like"}:
            return ["D", "A"]
        if speed_hint == "stop-like":
            return ["D", "A"]
        return ["A", "D"]

    def _format_nonrotation_candidate_lines(self, nonrotation_candidates: List[str]) -> str:
        descriptions = {
            "A": "A: 直線系（等速・加速・減速）",
            "D": "D: その他（停止・発進・その他）",
        }
        return "\n".join(descriptions[code] for code in nonrotation_candidates)

    def _extract_choice(self, text: str, allowed_codes: List[str]) -> Optional[str]:
        import re

        assistant_segments = re.findall(r'assistant\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if assistant_segments:
            response_text = assistant_segments[-1].strip()
        else:
            response_text = text.strip()

        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            strict_match = re.fullmatch(r'([A-Z])', last_line, re.IGNORECASE)
            if strict_match:
                code = strict_match.group(1).upper()
                if code in allowed_codes:
                    return code

        for code in allowed_codes:
            if re.search(rf'(?<![A-Z]){code}(?![A-Z])', response_text, re.IGNORECASE):
                return code

        label_matches = {
            "直線系": "A",
            "左回転系": "B",
            "右回転系": "C",
            "その他": "D",
            "回転系": "R",
        }
        for label, code in label_matches.items():
            if code in allowed_codes and label in response_text:
                return code

        fine_label = self._extract_action_label(text)
        if fine_label is not None:
            mapped = FINE_LABEL_TO_MACRO_GROUP.get(fine_label)
            if mapped in allowed_codes:
                return mapped
        return None

    def _macro_group_to_canonical_label(self, macro_group: str) -> int:
        return MACRO_GROUP_TO_CANONICAL_LABEL[macro_group]

    def _run_prompt_on_frames(
        self,
        frames_with_trajectory: List[Image.Image],
        prompt_text: str,
        max_new_tokens: int = 6,
    ) -> str:
        is_qwen = "qwen" in HERON_MODEL_ID.lower()

        if is_qwen:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": frame} for frame in frames_with_trajectory],
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt],
                images=frames_with_trajectory,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            text_prompt = prompt_text
            inputs = self.processor(
                text=prompt_text,
                images=frames_with_trajectory,
                return_tensors="pt"
            ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]
        return text_prompt, generated_text
    
    def predict_action(
        self,
        video_path: str,
        sensor_data: Dict[str, Any],
        start_time: float = 0.0,
        sample_id: Optional[int] = None,
        use_l2m: bool = None
    ) -> Optional[int]:
        """
        Predict action label from video and sensor data with trajectory visualization
        
        Args:
            video_path: Path to video file
            sensor_data: Dictionary containing sensor information
            start_time: Video start offset in seconds
            sample_id: Optional sample ID for saving trajectory frames
            use_l2m: Whether to use L2M+CoT pipeline (default: uses config.USE_L2M_COT)
        
        Returns:
            Predicted action label (0-10) or None if prediction fails
        """
        if (use_l2m or (use_l2m is None and USE_L2M_COT) or USE_VLM_DIRECT) and not self.is_loaded:
            self.load_model()
        
        # use_l2mが指定されていない場合はconfig設定を使用
        if use_l2m is None:
            use_l2m = USE_L2M_COT

        self._last_prediction_details = {
            "video_path": video_path,
            "sensor_data": dict(sensor_data),
            "start_time": start_time,
            "sample_id": sample_id,
            "mode": "sensor_only_rule_baseline" if USE_SENSOR_ONLY_BASELINE else ("l2m" if use_l2m else ("direct_vlm" if USE_VLM_DIRECT else "direct_deterministic"))
        }

        if USE_SENSOR_ONLY_BASELINE:
            return self._predict_sensor_only_baseline(
                sensor_data=sensor_data,
                sample_id=sample_id,
            )
        
        # L2M+CoTパイプラインを使用する場合
        if use_l2m:
            logger.info("Using L2M+CoT pipeline for prediction")
            return self._predict_with_l2m(video_path, sensor_data, start_time, sample_id)
        
        # Use multi-frame prediction with trajectory
        return self._predict_multi_frame_with_trajectory(video_path, sensor_data, start_time, sample_id)
    
    def _predict_with_l2m(
        self,
        video_path: str,
        sensor_data: Dict[str, Any],
        start_time: float = 0.0,
        sample_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Predict using L2M+CoT pipeline with trajectory visualization
        
        This method uses Least-to-Most Prompting combined with Chain-of-Thought
        to perform hierarchical analysis on trajectory-enhanced frames.
        """
        try:
            # heron_l2m_pipelineをインポート（遅延インポート）
            from heron_l2m_pipeline import L2MCoTPipeline
            
            # Extract 4 frames from video
            frames, frame_indices = self.extract_frames_from_video(
                video_path, 
                start_time=start_time,
                duration=5.0,
                num_frames=4
            )
            
            if not frames or len(frames) < 4:
                logger.error(f"Insufficient frames extracted for L2M: {len(frames)}/4")
                return None
            
            logger.info(f"[L2M+CoT] Extracted {len(frames)} frames at indices {frame_indices}")
            
            # Prepare model inputs as raw frames + trajectory-only summary
            model_frames = self.prepare_visual_inputs(frames, sensor_data, sample_id)

            speed_ms = sensor_data.get('speed', 0.0) / 3.6
            num_steps = 30
            speed_seq = np.full(num_steps, speed_ms, dtype=np.float32)
            yaw_rate_seq = np.full(num_steps, sensor_data.get('gyro_z', 0.0), dtype=np.float32)
            trajectory_3d = self.trajectory_visualizer.calculate_trajectory(speed_seq, yaw_rate_seq)
            _, valid_mask = self.trajectory_visualizer.project_3d_to_2d(trajectory_3d)
            visible_count = int(np.sum(valid_mask))
            
            # Create L2M pipeline
            pipeline = L2MCoTPipeline(self)
            
            # Run L2M analysis with trajectory-enhanced frames
            result = pipeline.analyze_with_l2m(frames_with_trajectory, sensor_data)
            
            # Extract final category
            final_category = result.get('final_category', 0)
            confidence = result.get('confidence', 0.5)
            
            logger.info(f"[L2M+CoT] *** FINAL RESULT: category={final_category}, confidence={confidence:.2f} ***")
            logger.info(f"[L2M+CoT] Level 1: {result['level1'].get('road_shape', 'N/A')}, {result['level1'].get('trajectory_relation', 'N/A')}")
            logger.info(f"[L2M+CoT] Level 2: {result['level2'].get('acceleration_cause', 'N/A')}, {result['level2'].get('speed_trend', 'N/A')}")
            logger.info(f"[L2M+CoT] Level 3: {result['level3'].get('category_name', 'N/A')}")
            logger.info(f"[L2M+CoT] *** RETURNING: {final_category} ***")
            
            return final_category
            
        except Exception as e:
            logger.error(f"L2M+CoT prediction FAILED with exception: {e}", exc_info=True)
            # フォールバック: 通常の推論を試す
            logger.warning("!!! FALLBACK to standard prediction !!!")
            return self._predict_multi_frame_with_trajectory(video_path, sensor_data, start_time, sample_id)
    
    def _predict_multi_frame_with_trajectory(
        self,
        video_path: str,
        sensor_data: Dict[str, Any],
        start_time: float = 0.0,
        sample_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Predict using multiple frames with trajectory visualization
        """
        try:
            # Extract 4 frames from video
            frames, frame_indices = self.extract_frames_from_video(
                video_path, 
                start_time=start_time,
                duration=5.0,
                num_frames=4  # Always use 4 frames
            )

            self._last_prediction_details = {
                **(self._last_prediction_details or {}),
                "frame_indices": frame_indices,
            }
            
            if not frames or len(frames) < 4:
                logger.error(f"Insufficient frames extracted: {len(frames)}/4")
                self._last_prediction_details = {
                    **(self._last_prediction_details or {}),
                    "num_frames": len(frames),
                    "error": f"Insufficient frames extracted: {len(frames)}/4",
                }
                return None
            
            logger.info(f"[Multi-frame with trajectory] Using {len(frames)} frames at indices {frame_indices}")
            
            # Prepare model inputs as raw frames + trajectory summary
            model_frames = self.prepare_visual_inputs(frames, sensor_data, sample_id)

            hints = self._derive_event_hints(
                speed=float(sensor_data.get('speed', 0.0)),
                acc_x=float(sensor_data.get('acc_x', 0.0)),
                gyro_z=float(sensor_data.get('gyro_z', 0.0)),
            )
            visible_points = int((self._last_prediction_details or {}).get("visible_trajectory_points", 0))
            stage_traces: List[Dict[str, Any]] = []
            final_prompt_text: Optional[str] = None
            final_generated_text: Optional[str] = None
            final_macro_candidates: List[str] = []
            macro_group: Optional[str] = None

            speed = sensor_data.get('speed', 0)
            acc_x = sensor_data.get('acc_x', 0)
            acc_y = sensor_data.get('acc_y', 0)
            acc_z = sensor_data.get('acc_z', 0)
            gyro_z = sensor_data.get('gyro_z', 0)
            latitude = sensor_data.get('latitude', 0)
            longitude = sensor_data.get('longitude', 0)

            trajectory_features = dict((self._last_prediction_details or {}).get("trajectory_features", {}))
            if not USE_VLM_DIRECT:
                macro_group, stage_traces = self._classify_macro_group_deterministic(
                    speed=float(speed),
                    acc_x=float(acc_x),
                    gyro_z=float(gyro_z),
                    visible_points=visible_points,
                    trajectory_features=trajectory_features,
                )
                final_macro_candidates = [macro_group]
                logger.info(f"Deterministic macro output: {macro_group}")
            else:
                stage1_choice, stage1_reason = self._determine_stage1_group(
                    speed=float(speed),
                    gyro_z=float(gyro_z),
                    visible_points=visible_points,
                    trajectory_features=trajectory_features,
                )
                stage_traces.append({
                    "stage": "stage1_deterministic",
                    "output": stage1_choice,
                    "reason": stage1_reason,
                })
                logger.info(f"Stage1 deterministic output: {stage1_choice} ({stage1_reason.get('rule')})")

                if stage1_choice == "R":
                    rotation_candidates = self._build_rotation_candidates(hints)
                    rotation_choice, rotation_reason = self._determine_rotation_direction(
                        gyro_z=float(gyro_z),
                        trajectory_features=trajectory_features,
                    )
                    stage_traces.append({
                        "stage": "stage2_rotation_deterministic",
                        "candidates": rotation_candidates,
                        "output": rotation_choice,
                        "reason": rotation_reason,
                    })
                    if rotation_choice == "AMB":
                        rotation_prompt = PROMPT_STAGE2_ROTATION_TEMPLATE.format(
                            speed=speed,
                            gyro_z=gyro_z,
                            latitude=latitude,
                            longitude=longitude,
                            candidate_lines=self._format_rotation_candidate_lines(rotation_candidates),
                        )
                        final_prompt_text, rotation_generated = self._run_prompt_on_frames(
                            model_frames,
                            rotation_prompt,
                            max_new_tokens=min(MAX_NEW_TOKENS_STANDARD, 4),
                        )
                        logger.info(f"Stage2 rotation output: {rotation_generated}")
                        rotation_choice = self._extract_choice(rotation_generated, rotation_candidates)
                        final_generated_text = rotation_generated
                        stage_traces.append({
                            "stage": "stage2_rotation_vlm",
                            "candidates": rotation_candidates,
                            "prompt_text": rotation_prompt,
                            "generated_text": rotation_generated,
                            "output": rotation_choice,
                        })
                    final_macro_candidates = ["B", "C"]
                    if rotation_choice == "L":
                        macro_group = "B"
                    elif rotation_choice == "R":
                        macro_group = "C"
                    else:
                        macro_group = "B" if float(trajectory_features.get("end_lateral_offset_m", 0.0)) >= 0 else "C"
                        stage_traces.append({
                            "stage": "stage2_rotation_fallback",
                            "reason": "failed_to_resolve_rotation_direction",
                            "output": macro_group,
                        })
                elif stage1_choice == "O":
                    final_macro_candidates = ["D"]
                    macro_group = "D"
                else:
                    final_macro_candidates = ["A"]
                    macro_group = "A"

            if final_generated_text:
                logger.info(f"Model output: {final_generated_text}")

            fine_label_reason: Optional[Dict[str, Any]] = None
            if macro_group:
                if USE_VLM_DIRECT:
                    action_label = self._macro_group_to_canonical_label(macro_group)
                else:
                    action_label, fine_label_reason = self._determine_fine_label_from_macro(
                        macro_group=macro_group,
                        speed=float(speed),
                        acc_x=float(acc_x),
                        gyro_z=float(gyro_z),
                        visible_points=visible_points,
                        trajectory_features=trajectory_features,
                    )
                    stage_traces.append({
                        "stage": "fine_label_deterministic",
                        "output": action_label,
                        "reason": fine_label_reason,
                    })
            else:
                action_label = None

            self._last_prediction_details = {
                **(self._last_prediction_details or {}),
                "macro_prediction_code": macro_group,
                "macro_prediction_label": MACRO_GROUP_LABELS.get(macro_group) if macro_group else None,
                "extracted_label": action_label,
                "prompt_text": final_prompt_text,
                "generated_text": final_generated_text,
                "num_frames": len(model_frames),
                "speed_event_hint": hints["speed_event_text"],
                "turn_event_hint": hints["turn_event_text"],
                "candidate_labels": [self._macro_group_to_canonical_label(code) for code in final_macro_candidates] if final_macro_candidates else None,
                "macro_candidate_codes": final_macro_candidates,
                "macro_candidate_labels": [MACRO_GROUP_LABELS[c] for c in final_macro_candidates] if final_macro_candidates else None,
                "stage_traces": stage_traces,
                "trajectory_features": trajectory_features,
                "stage1_reason": next((st.get("reason") for st in stage_traces if st.get("stage") == "stage1_deterministic"), None),
                "fine_label_reason": fine_label_reason,
            }
            
            if action_label is not None:
                self._last_prediction_details = {
                    **(self._last_prediction_details or {}),
                    "predicted_label": action_label,
                    "error": None,
                }
                logger.info(f"Predicted action label: {action_label}")
            else:
                self._last_prediction_details = {
                    **(self._last_prediction_details or {}),
                    "error": "Failed to extract macro motion group from model output",
                }
                logger.warning("Failed to extract macro motion group from model output")
            
            return action_label
            
        except Exception as e:
            self._last_prediction_details = {
                **(self._last_prediction_details or {}),
                "error": str(e)
            }
            logger.error(f"Error in multi-frame trajectory prediction: {e}", exc_info=True)
            return None
    
    def _extract_action_label(self, text: str) -> Optional[int]:
        """
        Extract action label from model's text output
        
        Args:
            text: Model generated text
        
        Returns:
            Action label (0-10) or None if extraction fails
        """
        import re

        # Prefer the final assistant segment. Qwen outputs often look like:
        # system ... user ... assistant\n1
        assistant_segments = re.findall(r'assistant\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if assistant_segments:
            response_text = assistant_segments[-1].strip()
        else:
            response_text = text.strip()

        # First, try the last non-empty line as a strict single-label answer.
        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            strict_match = re.fullmatch(r'(10|[0-9])', last_line)
            if strict_match:
                return int(strict_match.group(1))

        # Next, prefer the last standalone label token in the assistant response.
        standalone_numbers = re.findall(r'(?<!\d)(10|[0-9])(?!\d)', response_text)
        if standalone_numbers:
            action_label = int(standalone_numbers[-1])
            if 0 <= action_label <= 10:
                return action_label

        # Fallback for slightly more verbose outputs.
        patterns = [
            r'action[:\s]+(10|[0-9])',
            r'label[:\s]+(10|[0-9])',
            r'prediction[:\s]+(10|[0-9])',
            r'class[:\s]+(10|[0-9])',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_text.lower())
            if matches:
                action_label = int(matches[-1])
                if 0 <= action_label <= 10:
                    return action_label

        return None
    
    def predict_batch(
        self,
        samples: List[Dict[str, Any]],
        video_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple samples
        
        Args:
            samples: List of sample dictionaries with video_path, sensor_data, sample_id
            video_dir: Base directory for video files
        
        Returns:
            List of results with predictions
        """
        results = []
        
        for sample in samples:
            sample_id = sample.get('sample_id')
            video_path = sample.get('video_path')
            sensor_data = sample.get('sensor_data', {})
            start_time = sample.get('start_time', 0.0)
            
            logger.info(f"\nProcessing sample {sample_id}...")
            
            try:
                prediction = self.predict_action(
                    video_path=video_path,
                    sensor_data=sensor_data,
                    start_time=start_time,
                    sample_id=sample_id
                )
                
                results.append({
                    'sample_id': sample_id,
                    'prediction': prediction,
                    'success': prediction is not None
                })
                
            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {e}")
                results.append({
                    'sample_id': sample_id,
                    'prediction': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results


if __name__ == "__main__":
    # Test the trajectory visualization
    logging.basicConfig(level=logging.INFO)
    
    annotator = HeronAnnotatorWithTrajectory(
        save_trajectory_frames=True,
        trajectory_output_dir="trajectory_frames_test"
    )
    
    # Test trajectory drawing without model loading
    test_frame = Image.new('RGB', (1920, 1080), color=(50, 50, 50))
    test_sensor_data = {
        'speed': 40.0,  # km/h
        'gyro_z': 0.02,  # rad/s
        'acc_x': 0.5,
        'acc_y': 0.0,
        'acc_z': 0.0,
        'latitude': 35.6812,
        'longitude': 139.7671,
    }
    
    frames_with_traj = annotator.draw_trajectory_on_frames(
        [test_frame] * 4,
        test_sensor_data,
        sample_id=999
    )
    
    print(f"Generated {len(frames_with_traj)} frames with trajectory")


# ============ Singleton Instance ============
_annotator_instance = None

def get_annotator() -> HeronAnnotatorWithTrajectory:
    """
    Get singleton instance of HeronAnnotatorWithTrajectory
    Automatically saves trajectory frames to 'auto_annotation_trajectory_frames' directory
    """
    global _annotator_instance
    if _annotator_instance is None:
        _annotator_instance = HeronAnnotatorWithTrajectory(
            save_trajectory_frames=True,
            trajectory_output_dir="auto_annotation_trajectory_frames"
        )
    return _annotator_instance

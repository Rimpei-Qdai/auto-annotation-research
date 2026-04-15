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
from vlm_runtime import VLMGenerationRuntime

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
    PROMPT_TEMPLATE,
    ENABLE_FLASH_ATTENTION,
    USE_L2M_COT
)

logger = logging.getLogger(__name__)


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
        self._last_l2m_result = None  # predict_action_with_details で参照
        self._generation_runtime = None
        
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
    def generation_runtime(self) -> VLMGenerationRuntime:
        if self._generation_runtime is None:
            raise RuntimeError("Generation runtime is not initialized. Call load_model() first.")
        return self._generation_runtime

    @property
    def torch_dtype(self):
        """Resolve config dtype strings into torch.dtype values."""
        if isinstance(TORCH_DTYPE, torch.dtype):
            return TORCH_DTYPE

        if isinstance(TORCH_DTYPE, str):
            normalized = TORCH_DTYPE.strip().lower()
            if hasattr(torch, normalized):
                resolved = getattr(torch, normalized)
                if isinstance(resolved, torch.dtype):
                    return resolved

        logger.warning("Unsupported TORCH_DTYPE=%r. Falling back to torch.float16.", TORCH_DTYPE)
        return torch.float16

    @property
    def uses_device_map(self) -> bool:
        if self._generation_runtime is not None:
            return self._generation_runtime.uses_device_map
        device_map = getattr(self._model, "hf_device_map", None)
        return isinstance(device_map, dict) and len(device_map) > 0

    @property
    def generation_device(self) -> Optional[torch.device]:
        if self._generation_runtime is not None:
            return self._generation_runtime.generation_device
        return self._device

    def prepare_inputs_for_generation(self, inputs):
        return self.generation_runtime.prepare_inputs_for_generation(inputs)

    def _collect_tensor_devices(self, value):
        return self.generation_runtime._collect_tensor_devices(value)

    def _collect_model_devices(self):
        if self._generation_runtime is None:
            return set()
        return self._generation_runtime.collect_model_devices()
    
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
                self._device = torch.device("cuda:0")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self._device = torch.device("cpu")
                logger.info("Using CPU")
            
            # Check if Qwen2-VL model
            is_qwen = "qwen" in HERON_MODEL_ID.lower()
            load_with_explicit_device = self._device.type != "cpu"

            if load_with_explicit_device:
                logger.info(
                    "Loading model onto a single target device to avoid mixed CPU/CUDA placement during generation"
                )
            
            if is_qwen:
                # Qwen2-VL specific loading
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    HERON_MODEL_ID,
                    torch_dtype=self.torch_dtype,
                    device_map=None
                )
                self._processor = AutoProcessor.from_pretrained(HERON_MODEL_ID)
                logger.info("Qwen2-VL model loaded with multi-frame support")
            else:
                # Standard Heron model loading
                self._model = AutoModelForCausalLM.from_pretrained(
                    HERON_MODEL_ID,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True
                )
                self._processor = AutoProcessor.from_pretrained(
                    HERON_MODEL_ID,
                    trust_remote_code=True
                )
                self._is_heron = True
                logger.info(f"Heron model loaded with multi-frame support")
            
            if self._device is not None:
                self._model = self._model.to(self._device)
            self._generation_runtime = VLMGenerationRuntime(
                model=self._model,
                processor=self._processor,
                model_id=HERON_MODEL_ID,
                target_device=self._device,
            )
            self._generation_runtime.validate_model_placement()

            if self.uses_device_map:
                logger.info(f"Model uses Accelerate device_map: {self._model.hf_device_map}")

            logger.info(f"Generation inputs will be placed on: {self.generation_device}")
            
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
            frame_indices = self._compute_frame_indices(
                fps=fps,
                total_frames=total_frames,
                start_time=start_time,
                duration=duration,
                num_frames=num_frames,
            )
            
            frames = []
            actual_indices = []
            for frame_idx in frame_indices:
                actual_idx, frame = self._read_frame_with_fallback(cap, frame_idx, total_frames)

                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                    actual_indices.append(actual_idx)
                else:
                    logger.warning(f"Failed to read frame {frame_idx}")
            
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames, actual_indices
            
        finally:
            cap.release()

    def _compute_frame_indices(
        self,
        *,
        fps: float,
        total_frames: int,
        start_time: float,
        duration: float,
        num_frames: int,
    ) -> List[int]:
        if total_frames <= 0:
            return []
        if fps <= 0:
            return [0] * num_frames

        video_duration = total_frames / fps
        safe_duration = max(duration, num_frames / fps)
        window_end = min(max(start_time, 0.0) + safe_duration, video_duration)
        window_start = max(0.0, window_end - safe_duration)

        start_frame = max(0, min(int(window_start * fps), total_frames - 1))
        end_frame = max(start_frame + 1, min(int(window_end * fps), total_frames))
        base_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()

        ordered_unique: List[int] = []
        seen = set()
        for frame_idx in base_indices:
            clamped = max(0, min(int(frame_idx), total_frames - 1))
            if clamped not in seen:
                seen.add(clamped)
                ordered_unique.append(clamped)

        radius = 1
        while len(ordered_unique) < num_frames and len(seen) < total_frames:
            added = False
            for base_idx in base_indices:
                for offset in (-radius, radius):
                    candidate = base_idx + offset
                    if 0 <= candidate < total_frames and candidate not in seen:
                        seen.add(candidate)
                        ordered_unique.append(candidate)
                        added = True
                        if len(ordered_unique) >= num_frames:
                            break
                if len(ordered_unique) >= num_frames:
                    break
            if not added:
                break
            radius += 1

        ordered_unique.sort()
        if not ordered_unique:
            ordered_unique = [start_frame]
        while len(ordered_unique) < num_frames:
            ordered_unique.append(ordered_unique[-1])

        return ordered_unique[:num_frames]

    def _read_frame_with_fallback(
        self,
        cap: cv2.VideoCapture,
        frame_idx: int,
        total_frames: int,
        max_offset: int = 2,
    ) -> Tuple[int, Optional[np.ndarray]]:
        candidate_indices = [frame_idx]
        for offset in range(1, max_offset + 1):
            candidate_indices.extend([frame_idx - offset, frame_idx + offset])

        for candidate_idx in candidate_indices:
            if not (0 <= candidate_idx < total_frames):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
            ret, frame = cap.read()
            if ret:
                return candidate_idx, frame

        return frame_idx, None
    
    def draw_trajectory_on_frames(
        self,
        frames: List[Image.Image],
        sensor_data: Dict[str, Any],
        sample_id: Optional[int] = None,
        trajectory_bundle: Optional[Dict[str, Any]] = None,
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

        if trajectory_bundle is None:
            trajectory_bundle = self._build_trajectory_bundle(sensor_data)

        speed = float(sensor_data.get('speed', 0.0) or 0.0)  # km/h
        gyro_z = float(sensor_data.get('gyro_z', 0.0) or 0.0)  # rad/s (yaw rate)
        trajectory_3d = trajectory_bundle["trajectory_3d"]
        image_points = trajectory_bundle["image_points"]
        valid_mask = trajectory_bundle["valid_mask"]

        visible_count = np.sum(valid_mask)
        logger.info(f"Trajectory: {visible_count}/{len(valid_mask)} points visible")
        
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
                thickness=3,
                point_radius=4
            )
            
            # Add sensor info overlay
            overlay_text = f"Speed: {speed:.1f} km/h | Yaw rate: {gyro_z:.4f} rad/s | Frame {frame_idx+1}/4"
            cv2.putText(
                result_frame,
                overlay_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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

    def _build_trajectory_bundle(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trajectory and deterministic red-line features from the same data used for drawing."""
        speed = float(sensor_data.get('speed', 0.0) or 0.0)  # km/h
        gyro_z = float(sensor_data.get('gyro_z', 0.0) or 0.0)  # rad/s

        speed_ms = speed / 3.6
        num_steps = 30
        speed_seq = np.full(num_steps, speed_ms, dtype=np.float32)
        yaw_rate_seq = np.full(num_steps, gyro_z, dtype=np.float32)

        trajectory_3d = self.trajectory_visualizer.calculate_trajectory(speed_seq, yaw_rate_seq)
        image_points, valid_mask = self.trajectory_visualizer.project_3d_to_2d(trajectory_3d)
        trajectory_features = self.trajectory_visualizer.extract_trajectory_features(
            trajectory_3d=trajectory_3d,
            image_points=image_points,
            valid_mask=valid_mask,
            speed_kmh=speed,
        )
        logger.info("Deterministic trajectory features: %s", trajectory_features)

        return {
            "trajectory_3d": trajectory_3d,
            "image_points": image_points,
            "valid_mask": valid_mask,
            "features": trajectory_features,
        }

    def _augment_sensor_data_with_trajectory_features(self, sensor_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Attach deterministic red-line features to sensor_data for downstream reasoning."""
        trajectory_bundle = self._build_trajectory_bundle(sensor_data)
        enriched_sensor_data = dict(sensor_data)
        enriched_sensor_data.update(trajectory_bundle["features"])
        return enriched_sensor_data, trajectory_bundle
    
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
        if not self.is_loaded:
            self.load_model()
        
        # use_l2mが指定されていない場合はconfig設定を使用
        if use_l2m is None:
            use_l2m = USE_L2M_COT
        
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

            enriched_sensor_data, trajectory_bundle = self._augment_sensor_data_with_trajectory_features(sensor_data)

            # Draw trajectory on all 4 frames
            frames_with_trajectory = self.draw_trajectory_on_frames(
                frames,
                enriched_sensor_data,
                sample_id,
                trajectory_bundle=trajectory_bundle,
            )
            
            # Create L2M pipeline
            pipeline = L2MCoTPipeline(self.generation_runtime)
            
            # Run L2M analysis with trajectory-enhanced frames
            result = pipeline.analyze_with_l2m(frames_with_trajectory, enriched_sensor_data)

            # 詳細結果を保持（predict_action_with_details から参照できるよう）
            self._last_l2m_result = result

            if result.get('status') == 'failed' or result.get('final_category') is None:
                logger.warning(
                    "[L2M+CoT] Invalid result detected. Treating prediction as failure. error=%s",
                    result.get('error', 'unknown'),
                )
                return None
            
            # Extract final category
            final_category = result.get('final_category')
            confidence = result.get('confidence', 0.5)

            logger.info(f"[L2M+CoT] *** FINAL RESULT: category={final_category}, confidence={confidence:.2f} ***")
            logger.info(f"[L2M+CoT] Level 1: {result['level1'].get('road_shape', 'N/A')}, {result['level1'].get('trajectory_relation', 'N/A')}")
            logger.info(f"[L2M+CoT] Level 2: {result['level2'].get('acceleration_cause', 'N/A')}, {result['level2'].get('speed_trend', 'N/A')}")
            logger.info(f"[L2M+CoT] Level 3: {result['level3'].get('category_name', 'N/A')}")
            logger.info(f"[L2M+CoT] *** RETURNING: {final_category} ***")

            return final_category

        except Exception as e:
            logger.error(f"L2M+CoT prediction FAILED with exception: {e}", exc_info=True)
            logger.warning("!!! FALLBACK to standard prediction !!!")
            self._last_l2m_result = None
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
            
            if not frames or len(frames) < 4:
                logger.error(f"Insufficient frames extracted: {len(frames)}/4")
                return None
            
            logger.info(f"[Multi-frame with trajectory] Using {len(frames)} frames at indices {frame_indices}")

            enriched_sensor_data, trajectory_bundle = self._augment_sensor_data_with_trajectory_features(sensor_data)

            # Draw trajectory on all 4 frames
            frames_with_trajectory = self.draw_trajectory_on_frames(
                frames,
                enriched_sensor_data,
                sample_id,
                trajectory_bundle=trajectory_bundle,
            )
            
            # Format prompt with sensor data
            prompt_text = PROMPT_TEMPLATE.format(
                speed=enriched_sensor_data.get('speed', 0),
                acc_x=enriched_sensor_data.get('acc_x', 0),
                acc_y=enriched_sensor_data.get('acc_y', 0),
                acc_z=enriched_sensor_data.get('acc_z', 0),
                brake=enriched_sensor_data.get('brake', 0)
            )
            
            generated_text = self.generation_runtime.generate_text(
                frames_with_trajectory,
                prompt_text,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
            logger.info(f"Model output: {generated_text}")
            
            # Extract action label from response
            action_label = self._extract_action_label(generated_text)
            
            if action_label is not None:
                logger.info(f"Predicted action label: {action_label}")
            else:
                logger.warning("Failed to extract action label from model output")
            
            return action_label
            
        except Exception as e:
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
        
        # Extract only the assistant's response (after "assistant" marker)
        assistant_match = re.search(r'assistant\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if assistant_match:
            response_text = assistant_match.group(1).strip()
        else:
            # Fallback: use full text if no assistant marker found
            response_text = text
        
        # Look for patterns like "action: 2", "label: 5", or just a number
        patterns = [
            r'action[:\s]+(\d+)',
            r'label[:\s]+(\d+)',
            r'prediction[:\s]+(\d+)',
            r'class[:\s]+(\d+)',
            r'(\d+)'  # Last resort: any digit
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text.lower())
            if match:
                action_label = int(match.group(1))
                if 0 <= action_label <= 10:
                    return action_label
        
        return None
    
    def predict_action_with_details(
        self,
        video_path: str,
        sensor_data: Dict[str, Any],
        start_time: float = 0.0,
        sample_id: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """
        predict_action と同じだが、L2M の中間推論結果も返す。

        Returns:
            (predicted_label, l2m_details) のタプル。
            l2m_details は level1/level2/level3 の辞書。L2M未使用時は None。
        """
        self._last_l2m_result = None
        label = self.predict_action(video_path, sensor_data, start_time, sample_id)
        return label, self._last_l2m_result

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
        'brake': 0
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

# heron_model_with_trajectory.py
"""
Heron VLM Model Manager with Trajectory Visualization
Supports trajectory drawing on 4 frames for auto-annotation
"""

import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from datetime import datetime

# Import trajectory visualization
from visual_prompting import TrajectoryVisualizer
from driving_graph import MacroGraphVerifier

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
    STAGE1_PROMPT_TEMPLATE,
    STAGE2_PROMPT_TEMPLATE,
    ENABLE_FLASH_ATTENTION,
    USE_L2M_COT,
    MACRO_OUTPUT_TO_LABEL,
    MACRO_OUTPUT_NAMES,
)

logger = logging.getLogger(__name__)
REPRESENTATIVE_LABEL_TO_MACRO_NAME = {
    label_id: MACRO_OUTPUT_NAMES[macro_code]
    for macro_code, label_id in MACRO_OUTPUT_TO_LABEL.items()
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
        self.last_prediction_details = {}
        
        # Trajectory visualization settings
        self.save_trajectory_frames = save_trajectory_frames
        self.trajectory_output_dir = trajectory_output_dir
        self.trajectory_visualizer = None
        self.graph_verifier = MacroGraphVerifier()
        
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
            
            # Check if Qwen-VL model family
            is_qwen = "qwen" in HERON_MODEL_ID.lower()
            
            if is_qwen:
                # Qwen-VL family specific loading
                if "qwen3-vl" in HERON_MODEL_ID.lower():
                    try:
                        from transformers import Qwen3VLForConditionalGeneration
                    except ImportError as e:
                        raise RuntimeError(
                            "Qwen3-VL requires a transformers version with "
                            "Qwen3VLForConditionalGeneration support "
                            "(recommended: >=4.57.0)."
                        ) from e
                    qwen_loader_cls = Qwen3VLForConditionalGeneration
                    qwen_family_name = "Qwen3-VL"
                else:
                    from transformers import Qwen2VLForConditionalGeneration
                    qwen_loader_cls = Qwen2VLForConditionalGeneration
                    qwen_family_name = "Qwen2-VL"

                self._model = qwen_loader_cls.from_pretrained(
                    HERON_MODEL_ID,
                    torch_dtype=TORCH_DTYPE,
                    device_map="auto" if USE_GPU else None
                )
                self._processor = AutoProcessor.from_pretrained(HERON_MODEL_ID)
                logger.info(f"{qwen_family_name} model loaded with multi-frame support (memory optimized)")
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

    def _build_trajectory_geometry(
        self,
        sensor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build 3-second trajectory and summary stats from current sensor values."""
        speed = sensor_data.get('speed', 0.0)
        gyro_z = sensor_data.get('gyro_z', 0.0)
        speed_ms = speed / 3.6

        num_steps = 30
        speed_seq = np.full(num_steps, speed_ms, dtype=np.float32)
        yaw_rate_seq = np.full(num_steps, gyro_z, dtype=np.float32)
        trajectory_3d = self.trajectory_visualizer.calculate_trajectory(speed_seq, yaw_rate_seq)
        image_points, valid_mask = self.trajectory_visualizer.project_3d_to_2d(trajectory_3d)

        return {
            "speed": speed,
            "gyro_z": gyro_z,
            "trajectory_3d": trajectory_3d,
            "image_points": image_points,
            "valid_mask": valid_mask,
            "visible_count": int(np.sum(valid_mask)),
        }

    def _render_trajectory_summary(self, trajectory_3d: np.ndarray) -> Image.Image:
        """Render a clean bird's-eye summary image showing only trajectory geometry."""
        width, height = 720, 720
        margin = 60
        center_x = width // 2
        origin_y = height - margin
        max_forward_m = 35.0
        max_side_m = 12.0

        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Straight reference line
        draw.line(
            [(center_x, origin_y), (center_x, margin)],
            fill=(40, 180, 40),
            width=6,
        )

        points = []
        for x, y, _ in trajectory_3d:
            px = center_x + int((y / max_side_m) * (width * 0.32))
            py = origin_y - int((x / max_forward_m) * (height - 2 * margin))
            points.append((px, py))

        if len(points) == 1:
            draw.ellipse(
                [points[0][0] - 8, points[0][1] - 8, points[0][0] + 8, points[0][1] + 8],
                fill=(220, 30, 30),
                outline=(220, 30, 30),
            )
            return canvas

        # Halo then trajectory for high contrast
        if len(points) >= 2:
            draw.line(points, fill=(0, 0, 0), width=14)
            draw.line(points, fill=(220, 30, 30), width=8)

        # Start marker
        sx, sy = points[0]
        draw.ellipse([sx - 10, sy - 10, sx + 10, sy + 10], fill=(255, 255, 255), outline=(0, 0, 0), width=3)

        # End arrow marker
        ex, ey = points[-1]
        draw.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=(245, 180, 0), outline=(0, 0, 0), width=2)

        return canvas

    def prepare_visual_inputs(
        self,
        frames: List[Image.Image],
        sensor_data: Dict[str, Any],
        sample_id: Optional[int] = None
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """Prepare raw frames plus one clean trajectory summary image for VLM input."""
        geometry = self._build_trajectory_geometry(sensor_data)
        summary_image = self._render_trajectory_summary(geometry["trajectory_3d"])

        if self.save_trajectory_frames and sample_id is not None:
            os.makedirs(self.trajectory_output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(
                self.trajectory_output_dir,
                f"sample_{sample_id:03d}_trajectory_summary_{timestamp}.jpg",
            )
            summary_image.save(summary_path, quality=95)

        logger.info(
            f"Trajectory summary: {geometry['visible_count']}/{len(geometry['valid_mask'])} points visible"
        )
        model_frames = list(frames[:4]) + [summary_image]
        return model_frames, geometry

    def _run_prompt_on_frames(
        self,
        frames: List[Image.Image],
        prompt_text: str
    ) -> str:
        """Run a single prompt against the current VLM backend."""
        is_qwen = "qwen" in HERON_MODEL_ID.lower()

        if is_qwen:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": frame} for frame in frames],
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text_prompt],
                images=frames,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(
                text=prompt_text,
                images=frames,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        else:
            generated_ids = output_ids

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        return generated_text

    def _extract_choice(
        self,
        text: str,
        valid_choices: List[str],
        alias_to_choice: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Extract a stage choice from short generated text."""
        import re

        alias_to_choice = alias_to_choice or {}
        response_text = text.strip()
        if not response_text:
            return None

        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        candidates = []
        if lines:
            candidates.append(lines[-1])
        candidates.append(response_text[-80:])

        upper_choices = [choice.upper() for choice in valid_choices]
        for candidate in candidates:
            candidate_upper = candidate.upper().strip()
            if candidate_upper in upper_choices:
                return candidate_upper
            for choice in upper_choices:
                if re.search(rf'\b{re.escape(choice)}\b', candidate_upper):
                    return choice
            for alias, choice in alias_to_choice.items():
                if alias in candidate:
                    return choice.upper()

        return None

    def _apply_macro_graph(
        self,
        initial_macro_choice: str,
        sensor_data: Dict[str, Any],
        trajectory_features: Dict[str, Any],
        *,
        stage1_choice: str | None,
        stage2_choice: str | None,
    ) -> str:
        graph_result = self.graph_verifier.build(
            sensor_data,
            trajectory_features,
            stage1_choice=stage1_choice,
            stage2_choice=stage2_choice,
        )
        self.last_prediction_details["macro_graph"] = graph_result
        self.last_prediction_details["initial_macro_choice"] = initial_macro_choice

        strong_candidate = graph_result.get("strong_candidate")
        if strong_candidate and strong_candidate.get("macro_choice"):
            final_macro_choice = strong_candidate["macro_choice"]
            self.last_prediction_details["graph_override"] = final_macro_choice != initial_macro_choice
            self.last_prediction_details["final_macro_choice"] = final_macro_choice
            return final_macro_choice

        self.last_prediction_details["graph_override"] = False
        self.last_prediction_details["final_macro_choice"] = initial_macro_choice
        return initial_macro_choice
    
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
            
            # Draw trajectory on all 4 frames
            frames_with_trajectory = self.draw_trajectory_on_frames(frames, sensor_data, sample_id)
            
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
            self.last_prediction_details = {
                "mode": "qwen3_vl_stagewise",
                "sample_id": sample_id,
                "video_path": video_path,
                "start_time": start_time,
            }

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

            model_frames, geometry = self.prepare_visual_inputs(frames, sensor_data, sample_id)
            self.last_prediction_details["frame_indices"] = frame_indices
            self.last_prediction_details["trajectory_features"] = {
                "visible_count": geometry["visible_count"],
                "trajectory_points": len(geometry["trajectory_3d"]),
                "final_x_m": float(geometry["trajectory_3d"][-1][0]),
                "final_y_m": float(geometry["trajectory_3d"][-1][1]),
                "gyro_z": float(sensor_data.get('gyro_z', 0.0)),
                "speed": float(sensor_data.get('speed', 0.0)),
            }

            stage1_prompt = STAGE1_PROMPT_TEMPLATE.format(
                speed=sensor_data.get('speed', 0),
                acc_x=sensor_data.get('acc_x', 0),
                acc_y=sensor_data.get('acc_y', 0),
                acc_z=sensor_data.get('acc_z', 0),
                gyro_z=sensor_data.get('gyro_z', 0),
                brake=sensor_data.get('brake', 0)
            )
            stage1_generated = self._run_prompt_on_frames(model_frames, stage1_prompt)
            stage1_choice = self._extract_choice(
                stage1_generated,
                ["A", "N"],
                alias_to_choice={
                    "直線系": "A",
                    "左回転系": "N",
                    "右回転系": "N",
                    "その他": "N",
                },
            )

            self.last_prediction_details["stage1_prompt_text"] = stage1_prompt
            self.last_prediction_details["stage1_generated_text"] = stage1_generated
            self.last_prediction_details["stage1_choice"] = stage1_choice

            if stage1_choice == "A":
                final_macro_choice = self._apply_macro_graph(
                    "A",
                    sensor_data,
                    self.last_prediction_details["trajectory_features"],
                    stage1_choice=stage1_choice,
                    stage2_choice=None,
                )
                action_label = MACRO_OUTPUT_TO_LABEL[final_macro_choice]
                self.last_prediction_details["predicted_label"] = action_label
                logger.info(
                    f"Stage1 chose A -> Final macro {final_macro_choice} -> "
                    f"Predicted action label: {action_label} ({REPRESENTATIVE_LABEL_TO_MACRO_NAME[action_label]})"
                )
                return action_label

            if stage1_choice != "N":
                logger.warning(f"Failed to extract Stage1 choice from model output: {stage1_generated}")
                self.last_prediction_details["error"] = "stage1_parse_failed"
                return None

            stage2_prompt = STAGE2_PROMPT_TEMPLATE.format(
                speed=sensor_data.get('speed', 0),
                acc_x=sensor_data.get('acc_x', 0),
                acc_y=sensor_data.get('acc_y', 0),
                acc_z=sensor_data.get('acc_z', 0),
                gyro_z=sensor_data.get('gyro_z', 0),
                brake=sensor_data.get('brake', 0)
            )
            stage2_generated = self._run_prompt_on_frames(model_frames, stage2_prompt)
            stage2_choice = self._extract_choice(
                stage2_generated,
                ["B", "C", "D"],
                alias_to_choice={
                    "左回転系": "B",
                    "右回転系": "C",
                    "その他": "D",
                },
            )

            self.last_prediction_details["stage2_prompt_text"] = stage2_prompt
            self.last_prediction_details["stage2_generated_text"] = stage2_generated
            self.last_prediction_details["stage2_choice"] = stage2_choice

            if stage2_choice is None:
                logger.warning(f"Failed to extract Stage2 choice from model output: {stage2_generated}")
                self.last_prediction_details["error"] = "stage2_parse_failed"
                return None

            final_macro_choice = self._apply_macro_graph(
                stage2_choice,
                sensor_data,
                self.last_prediction_details["trajectory_features"],
                stage1_choice=stage1_choice,
                stage2_choice=stage2_choice,
            )
            action_label = MACRO_OUTPUT_TO_LABEL[final_macro_choice]
            self.last_prediction_details["predicted_label"] = action_label

            macro_name = REPRESENTATIVE_LABEL_TO_MACRO_NAME.get(action_label)
            if macro_name:
                logger.info(
                    f"Stage2 chose {stage2_choice} -> Final macro {final_macro_choice} -> "
                    f"Predicted action label: {action_label} ({macro_name})"
                )
            else:
                logger.info(f"Predicted action label: {action_label}")
            
            return action_label
            
        except Exception as e:
            self.last_prediction_details["error"] = str(e)
            logger.error(f"Error in multi-frame trajectory prediction: {e}", exc_info=True)
            return None
    
    def _extract_action_label(self, text: str) -> Optional[int]:
        """
        Extract action label from model's text output
        
        Args:
            text: Model generated text
        
        Returns:
            Representative 11-class label for the macro category or None
        """
        import re
        
        # Extract only the assistant's response (after "assistant" marker)
        assistant_match = re.search(r'assistant\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if assistant_match:
            response_text = assistant_match.group(1).strip()
        else:
            # Fallback: use only the tail to avoid re-matching prompt contents
            response_text = text[-200:].strip()

        normalized = response_text.strip()
        normalized_upper = normalized.upper()

        macro_patterns = [
            (r'^\s*A\s*$', "A"),
            (r'^\s*B\s*$', "B"),
            (r'^\s*C\s*$', "C"),
            (r'^\s*D\s*$', "D"),
            (r'\bA\b', "A"),
            (r'\bB\b', "B"),
            (r'\bC\b', "C"),
            (r'\bD\b', "D"),
        ]

        for pattern, macro_code in macro_patterns:
            if re.search(pattern, normalized_upper):
                return MACRO_OUTPUT_TO_LABEL[macro_code]

        japanese_aliases = [
            ("直線系", "A"),
            ("左回転系", "B"),
            ("右回転系", "C"),
            ("その他", "D"),
        ]

        for alias, macro_code in japanese_aliases:
            if alias in normalized:
                return MACRO_OUTPUT_TO_LABEL[macro_code]

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

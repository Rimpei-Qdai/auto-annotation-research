# visual_prompting.py
"""
Visual Prompting Module
車両軌道を映像上に可視化するモジュール
update_plan.mdセクション3「入力データ処理と視覚的プロンプトの数学的実装」に基づく実装
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """
    車両の予測軌道を計算し、映像に視覚的にオーバーレイするクラス
    """
    
    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        T_v2c: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            camera_matrix: カメラ内部パラメータ行列 K (3x3)
            dist_coeffs: 歪み係数 D (1x5)
            T_v2c: 車両座標系からカメラ座標系への変換行列 (4x4)
            image_size: 画像サイズ (width, height)
        """
        # 画像サイズを保存
        self.image_size = image_size if image_size is not None else (1280, 720)
        
        # デフォルトのカメラパラメータ（簡易設定）
        if camera_matrix is None:
            # 典型的な車載カメラの焦点距離と画像中心を想定
            self.camera_matrix = np.array([
                [800.0, 0.0, self.image_size[0] / 2.0],  # fx, 0, cx
                [0.0, 800.0, self.image_size[1] / 2.0],  # 0, fy, cy
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
        
        if dist_coeffs is None:
            # 歪み係数（放射状k1,k2,k3 + 接線p1,p2）
            self.dist_coeffs = np.zeros((1, 5), dtype=np.float32)
        else:
            self.dist_coeffs = dist_coeffs
        
        if T_v2c is None:
            # デフォルト変換行列（カメラが車両前方1.5m、高さ1.2m、下向き15度）
            pitch_angle = np.deg2rad(15)  # 下向き（正の角度）
            cos_p = np.cos(pitch_angle)
            sin_p = np.sin(pitch_angle)
            
            # 正しい座標変換行列
            R = np.array([
                [0, -1, 0],           # Camera X = -Vehicle Y
                [-sin_p, 0, -cos_p],  # Camera Y = 下向き成分
                [cos_p, 0, -sin_p]    # Camera Z = 前方成分
            ], dtype=np.float32)
            
            camera_position = np.array([1.5, 0.0, 1.2], dtype=np.float32)
            t = -R @ camera_position
            
            self.T_v2c = np.eye(4, dtype=np.float32)
            self.T_v2c[:3, :3] = R
            self.T_v2c[:3, 3] = t
        else:
            self.T_v2c = T_v2c
    
    def calculate_trajectory(
        self,
        speed_seq: np.ndarray,
        yaw_rate_seq: np.ndarray,
        dt: float = 0.1,
        initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """
        Bicycle Modelを用いて車両の未来軌道を計算
        
        Args:
            speed_seq: 速度シーケンス [m/s] shape: (N,)
            yaw_rate_seq: ヨーレートシーケンス [rad/s] shape: (N,)
            dt: サンプリング時間間隔 [s]
            initial_pose: 初期姿勢 (x, y, theta)
        
        Returns:
            trajectory_3d: 3D軌道点群 shape: (N+1, 3) [[x, y, z], ...]
        """
        x, y, theta = initial_pose
        trajectory_3d = [[x, y, 0.0]]  # 原点を含める（Z軸は平坦路面仮定）
        
        for v, w in zip(speed_seq, yaw_rate_seq):
            # Bicycle Modelによる位置更新
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += w * dt
            
            trajectory_3d.append([x, y, 0.0])
        
        return np.array(trajectory_3d, dtype=np.float32)
    
    def project_3d_to_2d(
        self,
        points_3d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        3D点群を2D画像平面に投影
        
        Args:
            points_3d: 3D点群 (車両座標系) shape: (N, 3)
        
        Returns:
            image_points: 2D画像座標 shape: (N, 2)
            valid_mask: 画像内に投影された点のマスク shape: (N,)
        """
        # 回転ベクトルと並進ベクトルを抽出（辞書とnumpy配列の両方に対応）
        if isinstance(self.T_v2c, dict):
            R = self.T_v2c['R']
            tvec = self.T_v2c['t']
        else:
            R = self.T_v2c[:3, :3]
            tvec = self.T_v2c[:3, 3].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        
        # 3D→2D投影
        image_points, _ = cv2.projectPoints(
            points_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        image_points = image_points.reshape(-1, 2)
        
        # カメラの後ろにある点を除外（Z座標が負の点）
        # カメラ座標系に変換して確認
        points_camera = (R @ points_3d.T).T + tvec.T
        in_front_of_camera = points_camera[:, 2] > 0.1  # カメラの前方0.1m以上
        
        # 画像内に収まっているかチェック
        valid_mask = (
            (image_points[:, 0] >= 0) & (image_points[:, 0] < self.image_size[0]) &
            (image_points[:, 1] >= 0) & (image_points[:, 1] < self.image_size[1]) &
            in_front_of_camera  # カメラの前方にある点のみ
        )
        
        return image_points, valid_mask
    
    def draw_trajectory_on_image(
        self,
        image: np.ndarray,
        trajectory_3d: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),  # BGR形式で赤
        thickness: int = 10,
        draw_reference_line: bool = True,
        point_radius: int = 10
    ) -> np.ndarray:
        """
        計算された3D軌道を画像上に点と線として描画
        速度が低い場合でも点が見えるように、まず点をプロットしてから線で結ぶ
        
        Args:
            image: 入力画像 (H, W, 3)
            trajectory_3d: 3D軌道点群 shape: (N, 3)
            color: 線の色（BGR）
            thickness: 線の太さ
            draw_reference_line: 緑の参照線（直進方向）も描画するか
            point_radius: 軌道点の半径
        
        Returns:
            描画後の画像
        """
        
        # 入力画像をコピー（元の画像を変更しない）
        output_image = image.copy()
        
        # 3D軌道を2D画像平面に投影
        image_points, valid_mask = self.project_3d_to_2d(trajectory_3d)
        
        num_valid = np.sum(valid_mask)
        
        # 停止付近では前進っぽい fallback を描かず、STOP cue を表示する
        if num_valid < 1:
            logger.info("Drawing stop badge fallback (no visible trajectory points)")
            self._draw_stop_badge(output_image)
            return output_image
        
        # 有効な点を取得
        valid_points = image_points[valid_mask].astype(np.int32)

        # 1. 時系列が見えるように、軌道を濃淡付きの線分として描画
        self._draw_temporal_trajectory(
            output_image,
            valid_points,
            base_thickness=thickness,
            point_radius=point_radius,
        )

        # 緑の参照線（車線中心や直進方向）を描画
        if draw_reference_line:
            # 簡易的に直進方向を緑線として描画（前方20mまで）
            straight_line_3d = np.array([
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0]
            ], dtype=np.float32)
            
            ref_points, ref_valid = self.project_3d_to_2d(straight_line_3d)
            
            if np.sum(ref_valid) == 2:
                pts_green = ref_points[ref_valid].astype(np.int32)
                self._draw_dashed_line(
                    output_image,
                    tuple(pts_green[0]),
                    tuple(pts_green[1]),
                    (90, 255, 120),
                    1,
                    dash_length=22,
                )

        focus_roi = self._compute_focus_roi(valid_points, output_image.shape[:2])
        if focus_roi is not None:
            self._draw_focus_inset(output_image, focus_roi)

        return output_image

    def render_trajectory_summary(
        self,
        trajectory_3d: np.ndarray,
        speed: float,
        acc_x: float,
        gyro_z: float,
        latitude: float,
        longitude: float,
        canvas_size: Tuple[int, int] = (960, 960),
    ) -> np.ndarray:
        """Render a pure trajectory-only summary image for VLM input."""
        width, height = canvas_size
        image = np.full((height, width, 3), 250, dtype=np.uint8)

        # Drawing region
        top = 70
        bottom = height - 70
        left = 90
        right = width - 90
        center_x = (left + right) // 2

        cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), -1)
        cv2.rectangle(image, (left, top), (right, bottom), (210, 210, 210), 2)

        # Straight reference line
        self._draw_dashed_line(
            image,
            (center_x, bottom - 20),
            (center_x, top + 24),
            (90, 200, 90),
            3,
            dash_length=28,
        )

        # Project BEV trajectory into a simple summary plane.
        points_xy = np.asarray(trajectory_3d[:, :2], dtype=np.float32)
        max_forward_m = max(18.0, float(np.max(points_xy[:, 0]) + 2.0))
        lateral_range_m = max(5.0, float(np.max(np.abs(points_xy[:, 1])) + 1.5))

        def to_canvas(pt: np.ndarray) -> Tuple[int, int]:
            x_m, y_m = float(pt[0]), float(pt[1])
            norm_x = np.clip(x_m / max_forward_m, 0.0, 1.0)
            norm_y = np.clip(y_m / lateral_range_m, -1.0, 1.0)
            px = int(center_x - norm_y * ((right - left) * 0.36))
            py = int(bottom - 22 - norm_x * ((bottom - top) - 44))
            return px, py

        canvas_points = np.array([to_canvas(pt) for pt in points_xy], dtype=np.int32)

        # Start marker
        start = tuple(canvas_points[0])
        cv2.circle(image, start, 18, (0, 0, 0), -1)
        cv2.circle(image, start, 15, (255, 255, 255), 3)
        cv2.circle(image, start, 10, (70, 140, 255), -1)

        forward_distance = float(points_xy[-1, 0])
        stop_like = speed < 1.0 or forward_distance < 0.8

        if stop_like:
            # Keep only a very short trajectory cue near the start for near-stationary cases.
            cv2.circle(image, start, 18, (0, 0, 0), -1)
            cv2.circle(image, start, 15, (255, 255, 255), 3)
            cv2.circle(image, start, 10, (255, 90, 90), -1)
        else:
            for idx in range(len(canvas_points) - 1):
                p1 = tuple(canvas_points[idx])
                p2 = tuple(canvas_points[idx + 1])
                progress = idx / max(1, len(canvas_points) - 2)
                color = (
                    255,
                    int(40 + 180 * progress),
                    int(20 + 25 * progress),
                )
                cv2.line(image, p1, p2, (0, 0, 0), 18, cv2.LINE_AA)
                cv2.line(image, p1, p2, color, 10, cv2.LINE_AA)

            if len(canvas_points) >= 2:
                cv2.arrowedLine(
                    image,
                    tuple(canvas_points[-2]),
                    tuple(canvas_points[-1]),
                    (0, 0, 0),
                    22,
                    cv2.LINE_AA,
                    tipLength=0.45,
                )
                cv2.arrowedLine(
                    image,
                    tuple(canvas_points[-2]),
                    tuple(canvas_points[-1]),
                    (255, 220, 0),
                    12,
                    cv2.LINE_AA,
                    tipLength=0.40,
                )

        return image

    def _draw_temporal_trajectory(
        self,
        image: np.ndarray,
        valid_points: np.ndarray,
        base_thickness: int,
        point_radius: int,
    ) -> None:
        """Draw the trajectory with temporal cues: gradient, ticks, start ring, and end arrow."""
        if len(valid_points) == 0:
            return

        # Draw start point as a white ring.
        cv2.circle(image, tuple(valid_points[0]), point_radius + 6, (0, 0, 0), -1)
        cv2.circle(image, tuple(valid_points[0]), point_radius + 4, (255, 255, 255), 3)
        cv2.circle(image, tuple(valid_points[0]), point_radius + 1, (255, 90, 90), -1)

        if len(valid_points) == 1:
            return

        segment_count = len(valid_points) - 1
        marker_indices = {
            max(1, int(round(segment_count / 3))),
            max(1, int(round((2 * segment_count) / 3))),
        }

        for idx in range(segment_count):
            p1 = tuple(valid_points[idx])
            p2 = tuple(valid_points[idx + 1])
            progress = idx / max(1, segment_count - 1)

            # Start with darker red and gradually brighten towards yellow.
            color = (
                255,
                int(40 + 180 * progress),
                int(20 + 25 * progress),
            )
            thickness = base_thickness + (3 if progress > 0.45 else 0)
            # Add a dark halo so the trajectory survives bright/complex road texture.
            cv2.line(image, p1, p2, (0, 0, 0), thickness + 6, cv2.LINE_AA)
            cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)

            cv2.circle(image, p1, point_radius + 4, (0, 0, 0), -1)
            cv2.circle(image, p1, point_radius, color, -1)

            if idx in marker_indices:
                cv2.circle(image, p1, point_radius + 7, (0, 0, 0), 2)
                cv2.circle(image, p1, point_radius + 5, (255, 255, 255), 3)
                cv2.circle(image, p1, point_radius + 2, (255, 255, 255), -1)

        end_start = tuple(valid_points[-2])
        end_point = tuple(valid_points[-1])
        cv2.arrowedLine(
            image,
            end_start,
            end_point,
            (0, 0, 0),
            base_thickness + 10,
            cv2.LINE_AA,
            tipLength=0.40
        )
        cv2.arrowedLine(
            image,
            end_start,
            end_point,
            (255, 220, 0),
            base_thickness + 4,
            cv2.LINE_AA,
            tipLength=0.36
        )
        cv2.circle(image, end_point, point_radius + 8, (0, 0, 0), -1)
        cv2.circle(image, end_point, point_radius + 4, (255, 220, 0), -1)
        cv2.circle(image, end_point, point_radius + 9, (255, 255, 255), 3)

    def _compute_focus_roi(
        self,
        valid_points: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute a crop around the trajectory so the model can see a zoomed view."""
        if len(valid_points) < 2:
            return None

        image_h, image_w = image_shape
        min_x = int(np.min(valid_points[:, 0]))
        max_x = int(np.max(valid_points[:, 0]))
        min_y = int(np.min(valid_points[:, 1]))
        max_y = int(np.max(valid_points[:, 1]))

        pad_x = max(80, int((max_x - min_x) * 0.8))
        pad_y = max(80, int((max_y - min_y) * 0.8))

        x1 = max(0, min_x - pad_x)
        x2 = min(image_w, max_x + pad_x)
        y1 = max(int(image_h * 0.45), min_y - pad_y)
        y2 = min(image_h, max_y + pad_y)

        if x2 - x1 < 120 or y2 - y1 < 120:
            return None

        return (x1, y1, x2, y2)

    def _draw_focus_inset(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> None:
        """Draw an enlarged crop of the trajectory region in the upper-right corner."""
        x1, y1, x2, y2 = roi
        roi_crop = image[y1:y2, x1:x2]
        if roi_crop.size == 0:
            return

        image_h, image_w = image.shape[:2]
        inset_w = int(image_w * 0.38)
        inset_h = int(image_h * 0.34)
        inset = cv2.resize(roi_crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)

        margin = 18
        panel_x1 = image_w - inset_w - margin
        panel_y1 = margin + 12
        panel_x2 = image_w - margin
        panel_y2 = panel_y1 + inset_h

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x1 - 8, panel_y1 - 34),
            (panel_x2 + 8, panel_y2 + 8),
            (12, 12, 12),
            -1
        )
        cv2.addWeighted(overlay, 0.50, image, 0.50, 0, image)
        image[panel_y1:panel_y2, panel_x1:panel_x2] = inset
        cv2.rectangle(image, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 255, 255), 2)
        cv2.putText(
            image,
            "zoomed trajectory",
            (panel_x1 + 10, panel_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Show the ROI on the original frame so the relation between views is clear.
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Add a simple scale cue in the inset instead of small textual legends.
        cue_y = panel_y2 - 22
        cue_x1 = panel_x1 + 18
        cue_x2 = cue_x1 + 70
        cv2.line(image, (cue_x1, cue_y), (cue_x2, cue_y), (0, 0, 0), 10, cv2.LINE_AA)
        cv2.line(image, (cue_x1, cue_y), (cue_x2, cue_y), (255, 90, 40), 6, cv2.LINE_AA)
        cv2.putText(
            image,
            "thick path cue",
            (cue_x2 + 12, cue_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    def _draw_dashed_line(
        self,
        image: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_length: int = 16,
    ) -> None:
        """Draw a dashed line between two points."""
        start_pt = np.array(start, dtype=np.float32)
        end_pt = np.array(end, dtype=np.float32)
        delta = end_pt - start_pt
        total_length = float(np.linalg.norm(delta))

        if total_length < 1.0:
            cv2.line(image, start, end, color, thickness)
            return

        direction = delta / total_length
        draw = True
        current_length = 0.0

        while current_length < total_length:
            segment_start = start_pt + direction * current_length
            next_length = min(total_length, current_length + dash_length)
            segment_end = start_pt + direction * next_length

            if draw:
                cv2.line(
                    image,
                    tuple(segment_start.astype(int)),
                    tuple(segment_end.astype(int)),
                    color,
                    thickness,
                    cv2.LINE_AA
                )

            draw = not draw
            current_length = next_length

    def _draw_stop_badge(self, image: np.ndarray) -> None:
        """Draw a large stop badge when projected motion is essentially absent."""
        h, w = image.shape[:2]
        cx = w // 2
        cy = int(h * 0.74)
        radius = max(44, min(w, h) // 14)

        overlay = image.copy()
        cv2.circle(overlay, (cx, cy), radius + 14, (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.42, image, 0.58, 0, image)
        cv2.circle(image, (cx, cy), radius + 8, (255, 255, 255), 4)
        cv2.circle(image, (cx, cy), radius, (0, 0, 255), -1)
        cv2.putText(
            image,
            "STOP",
            (cx - radius + 12, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    
    def visualize_trajectory(
        self,
        image: np.ndarray,
        speed_seq: np.ndarray,
        yaw_rate_seq: np.ndarray,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        統合関数：速度とヨーレートから軌道を計算し、画像に描画
        
        Args:
            image: 入力画像 (H, W, 3)
            speed_seq: 未来の速度シーケンス [m/s]
            yaw_rate_seq: 未来のヨーレートシーケンス [rad/s]
            dt: サンプリング時間間隔 [s]
        
        Returns:
            描画後の画像
        """
        # 1. 軌道計算
        trajectory_3d = self.calculate_trajectory(speed_seq, yaw_rate_seq, dt)
        
        # 2. 画像に描画
        output_image = self.draw_trajectory_on_image(
            image,
            trajectory_3d,
            color=(0, 0, 255),  # 赤い軌道
            thickness=3,
            draw_reference_line=True  # 緑の参照線も描画
        )
        
        return output_image


def extract_sensor_sequences(
    sensor_data: Dict[str, Any],
    prediction_horizon: float = 3.0,
    dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    センサーデータから速度とヨーレートのシーケンスを抽出
    
    Args:
        sensor_data: センサーデータ辞書
        prediction_horizon: 予測時間範囲 [s]
        dt: サンプリング間隔 [s]
    
    Returns:
        speed_seq: 速度シーケンス [m/s]
        yaw_rate_seq: ヨーレートシーケンス [rad/s]
    """
    n_steps = int(prediction_horizon / dt)
    
    # 現在の速度とヨーレートを取得（km/h→m/s変換）
    current_speed = sensor_data.get('speed', 0.0) / 3.6  # km/h → m/s
    current_yaw_rate = sensor_data.get('gyro_z', 0.0)  # rad/s
    
    # 簡易的に一定速度・一定角速度と仮定してシーケンスを生成
    # 実際には加速度を考慮した動的モデルを使用可能
    speed_seq = np.full(n_steps, current_speed, dtype=np.float32)
    yaw_rate_seq = np.full(n_steps, current_yaw_rate, dtype=np.float32)
    
    return speed_seq, yaw_rate_seq


def create_visual_prompt(
    image: np.ndarray,
    sensor_data: Dict[str, Any],
    visualizer: Optional[TrajectoryVisualizer] = None
) -> np.ndarray:
    """
    入力画像にVisual Promptingを適用
    
    Args:
        image: 入力画像
        sensor_data: センサーデータ
        visualizer: TrajectoryVisualizerインスタンス
    
    Returns:
        Visual Prompting適用後の画像
    """
    if visualizer is None:
        visualizer = TrajectoryVisualizer()
    
    # センサーデータから速度・ヨーレートシーケンスを抽出
    speed_seq, yaw_rate_seq = extract_sensor_sequences(sensor_data)
    
    # 軌道を可視化
    output_image = visualizer.visualize_trajectory(
        image,
        speed_seq,
        yaw_rate_seq
    )
    
    return output_image


# ===== 使用例 =====
if __name__ == "__main__":
    # テスト用の画像とセンサーデータ
    test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_image[:, :] = [100, 100, 100]  # グレー背景
    
    test_sensor = {
        'speed': 50.0,  # km/h
        'gyro_z': 0.05,  # rad/s (左カーブ)
        'acc_x': 0.0,
        'acc_y': 0.0,
        'acc_z': 0.0
    }
    
    # Visual Prompting適用
    visualizer = TrajectoryVisualizer()
    result_image = create_visual_prompt(test_image, test_sensor, visualizer)
    
    # 結果を保存
    cv2.imwrite("test_visual_prompt.png", result_image)
    print("Visual prompting test completed. Check test_visual_prompt.png")

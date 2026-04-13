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

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
    
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

    def extract_trajectory_features(
        self,
        trajectory_3d: np.ndarray,
        image_points: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
        speed_kmh: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        赤い軌道そのものから deterministic な特徴を抽出する。

        主に以下を返す:
        - 左右の曲がり方向
        - 曲がりの強さ
        - 横方向のずれ
        - 線の長さに基づく速度の見え方
        - Level 1 で使う cue / relation / shift
        """
        if trajectory_3d is None or len(trajectory_3d) == 0:
            return {
                "trajectory_feature_source": "PROJECTED_RED_TRAJECTORY",
                "trajectory_path_length_m": 0.0,
                "trajectory_visible_length_px": 0.0,
                "trajectory_visible_point_count": 0,
                "trajectory_point_like": True,
                "trajectory_lateral_offset_m": 0.0,
                "trajectory_forward_extent_m": 0.0,
                "trajectory_heading_delta_rad": 0.0,
                "trajectory_curvature_score": 0.0,
                "trajectory_length_state": "POINT",
                "trajectory_visual_speed_state": "STOPPED",
                "trajectory_motion_cue_deterministic": "STRAIGHT_CUE",
                "trajectory_relation_deterministic": "PARALLEL",
                "direction_change_deterministic": "STRAIGHT",
                "visual_shift_deterministic": "NO_SHIFT",
            }

        if image_points is None or valid_mask is None:
            image_points, valid_mask = self.project_3d_to_2d(trajectory_3d)

        diffs_xy = np.diff(trajectory_3d[:, :2], axis=0)
        segment_lengths_m = np.linalg.norm(diffs_xy, axis=1) if len(diffs_xy) else np.array([], dtype=np.float32)
        path_length_m = float(segment_lengths_m.sum())

        forward_extent_m = float(max(0.0, trajectory_3d[-1, 0] - trajectory_3d[0, 0]))
        lateral_offset_m = float(trajectory_3d[-1, 1] - trajectory_3d[0, 1])

        nonzero_segments = diffs_xy[np.linalg.norm(diffs_xy, axis=1) > 1e-4] if len(diffs_xy) else np.empty((0, 2))
        if len(nonzero_segments) >= 2:
            start_heading = float(np.arctan2(nonzero_segments[0, 1], nonzero_segments[0, 0]))
            end_heading = float(np.arctan2(nonzero_segments[-1, 1], nonzero_segments[-1, 0]))
            heading_delta_rad = self._wrap_angle(end_heading - start_heading)
        else:
            heading_delta_rad = 0.0

        curvature_score = abs(heading_delta_rad)
        valid_points = image_points[valid_mask].astype(np.float32) if np.any(valid_mask) else np.empty((0, 2), dtype=np.float32)
        if len(valid_points) >= 2:
            visible_length_px = float(np.linalg.norm(np.diff(valid_points, axis=0), axis=1).sum())
        else:
            visible_length_px = 0.0

        point_like = (
            len(valid_points) <= 1
            or visible_length_px < 18.0
            or path_length_m < 0.8
            or (speed_kmh is not None and speed_kmh < 1.0)
        )

        if point_like:
            length_state = "POINT"
            visual_speed_state = "STOPPED"
        elif path_length_m < 3.0 or visible_length_px < 70.0:
            length_state = "SHORT"
            visual_speed_state = "SLOW"
        elif path_length_m < 9.0 or visible_length_px < 180.0:
            length_state = "MEDIUM"
            visual_speed_state = "MEDIUM"
        else:
            length_state = "LONG"
            visual_speed_state = "FAST"

        if lateral_offset_m >= 0.45:
            visual_shift = "SHIFT_LEFT"
        elif lateral_offset_m <= -0.45:
            visual_shift = "SHIFT_RIGHT"
        else:
            visual_shift = "NO_SHIFT"

        abs_lateral_offset_m = abs(lateral_offset_m)
        abs_heading_delta_rad = abs(heading_delta_rad)

        if point_like:
            motion_cue = "STRAIGHT_CUE"
            direction_change = "STRAIGHT"
            trajectory_relation = "PARALLEL"
        elif abs_heading_delta_rad >= 0.18 and abs_lateral_offset_m >= 0.45:
            motion_cue = "LEFT_TURN_CUE" if lateral_offset_m > 0 else "RIGHT_TURN_CUE"
            direction_change = "TURNING"
            if abs_lateral_offset_m >= 0.75:
                trajectory_relation = "CROSSING_LEFT" if lateral_offset_m > 0 else "CROSSING_RIGHT"
            else:
                trajectory_relation = "PARALLEL"
        elif abs_lateral_offset_m >= 0.75 and abs_heading_delta_rad < 0.18:
            motion_cue = "LEFT_LANE_CHANGE_CUE" if lateral_offset_m > 0 else "RIGHT_LANE_CHANGE_CUE"
            direction_change = "STRAIGHT"
            trajectory_relation = "CROSSING_LEFT" if lateral_offset_m > 0 else "CROSSING_RIGHT"
        elif abs_lateral_offset_m < 0.35 and abs_heading_delta_rad < 0.12:
            motion_cue = "STRAIGHT_CUE"
            direction_change = "STRAIGHT"
            trajectory_relation = "PARALLEL"
        else:
            motion_cue = "AMBIGUOUS"
            direction_change = "TURNING" if abs_heading_delta_rad >= 0.18 else "STRAIGHT"
            if abs_lateral_offset_m >= 0.75:
                trajectory_relation = "CROSSING_LEFT" if lateral_offset_m > 0 else "CROSSING_RIGHT"
            else:
                trajectory_relation = "PARALLEL"

        return {
            "trajectory_feature_source": "PROJECTED_RED_TRAJECTORY",
            "trajectory_path_length_m": round(path_length_m, 3),
            "trajectory_visible_length_px": round(visible_length_px, 1),
            "trajectory_visible_point_count": int(len(valid_points)),
            "trajectory_point_like": bool(point_like),
            "trajectory_lateral_offset_m": round(lateral_offset_m, 3),
            "trajectory_forward_extent_m": round(forward_extent_m, 3),
            "trajectory_heading_delta_rad": round(heading_delta_rad, 3),
            "trajectory_curvature_score": round(curvature_score, 3),
            "trajectory_length_state": length_state,
            "trajectory_visual_speed_state": visual_speed_state,
            "trajectory_motion_cue_deterministic": motion_cue,
            "trajectory_relation_deterministic": trajectory_relation,
            "direction_change_deterministic": direction_change,
            "visual_shift_deterministic": visual_shift,
        }
    
    def draw_trajectory_on_image(
        self,
        image: np.ndarray,
        trajectory_3d: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),  # BGR形式で赤
        thickness: int = 3,
        draw_reference_line: bool = True,
        point_radius: int = 4
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
        
        # 速度が0で軌道点が投影できない場合は、車両の少し前方の位置を描画
        if num_valid < 1:
            # 車両の5m前方の点を描画（起点の代わり）
            fallback_point = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
            fallback_image_points, fallback_mask = self.project_3d_to_2d(fallback_point)
            
            if np.sum(fallback_mask) > 0:
                logger.info("Drawing fallback point at 5m forward (low/zero speed)")
                image_points = fallback_image_points
                valid_mask = fallback_mask
                num_valid = 1
            else:
                logger.warning("No valid points to draw trajectory (even fallback failed)")
                return output_image
        
        # 有効な点を取得
        valid_points = image_points[valid_mask].astype(np.int32)
        
        # 1. まず各点を円でプロット（速度が低い場合でも見える）
        for point in valid_points:
            cv2.circle(
                output_image,
                tuple(point),
                point_radius,
                color,
                -1  # 塗りつぶし
            )
        
        # 2. 点が2つ以上ある場合は線で結ぶ
        if num_valid >= 2:
            cv2.polylines(
                output_image,
                [valid_points],
                isClosed=False,
                color=color,
                thickness=max(1, thickness - 1)  # 線は少し細めに
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
                cv2.line(
                    output_image,
                    tuple(pts_green[0]),
                    tuple(pts_green[1]),
                    (0, 255, 0),  # 緑色
                    2
                )
        
        return output_image
    
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

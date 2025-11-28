"""
ToonComposer スタイルの高品質フレーム補間エンジン
エッジ保存、色保持、アニメーション対応の補間を実現
"""

import numpy as np
from typing import List, Tuple
from scipy.ndimage import gaussian_filter, median_filter


class ToonStyleInterpolator:
    """ToonComposer のような高品質なアニメーション中割を生成"""
    
    def __init__(self):
        """初期化"""
        self.edge_threshold = 0.1
        self.smoothing_sigma = 1.5
        
    def interpolate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int,
        preserve_edges: bool = True,
        use_color_correction: bool = True
    ) -> List[np.ndarray]:
        """
        ToonComposer スタイルの補間
        
        Args:
            frame1, frame2: 入力フレーム（uint8 RGB）
            num_frames: 生成する中割フレーム数
            preserve_edges: エッジ保存を使用するかどうか
            use_color_correction: 色補正を使用するかどうか
        
        Returns:
            補間フレームのリスト
        """
        try:
            frame1_float = frame1.astype(np.float32) / 255.0
            frame2_float = frame2.astype(np.float32) / 255.0
            
            interpolated_frames = []
            
            for i in range(1, num_frames + 1):
                t = i / (num_frames + 1)
                
                try:
                    # 基本的な補間
                    blended = (1 - t) * frame1_float + t * frame2_float
                    
                    # エッジ検出と保存
                    if preserve_edges:
                        edge_mask = self._detect_edges(frame1, frame2)
                        blended = self._apply_edge_preservation(
                            frame1_float, frame2_float, blended, edge_mask, t
                        )
                    
                    # 色域補正
                    if use_color_correction:
                        blended = self._apply_color_correction(frame1_float, frame2_float, blended, t)
                    
                    # アニメーション平滑化
                    blended = self._apply_animation_smoothing(blended)
                    
                    # uint8 に変換
                    result = np.clip(blended * 255, 0, 255).astype(np.uint8)
                    interpolated_frames.append(result)
                    
                except Exception as e:
                    print(f"⚠ Frame {i} interpolation error: {e}")
                    # フレーム生成失敗時は単純補間
                    simple = np.clip((1 - t) * frame1 + t * frame2, 0, 255).astype(np.uint8)
                    interpolated_frames.append(simple)
            
            return interpolated_frames
            
        except Exception as e:
            print(f"❌ Interpolation failed: {e}")
            # 最終フォールバック
            interpolated_frames = []
            for i in range(1, num_frames + 1):
                t = i / (num_frames + 1)
                frame = np.clip((1 - t) * frame1 + t * frame2, 0, 255).astype(np.uint8)
                interpolated_frames.append(frame)
            return interpolated_frames
    
    @staticmethod
    def _detect_edges(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        エッジ検出（Sobelフィルタ）
        
        Returns:
            エッジマスク [H, W]
        """
        try:
            from scipy.ndimage import sobel
            
            # グレースケール変換
            gray1 = np.dot(frame1[..., :3], [0.299, 0.587, 0.114]) if frame1.ndim == 3 else frame1
            gray2 = np.dot(frame2[..., :3], [0.299, 0.587, 0.114]) if frame2.ndim == 3 else frame2
            
            # Sobelフィルタ
            sx1 = sobel(gray1, axis=1)
            sy1 = sobel(gray1, axis=0)
            edge1 = np.sqrt(sx1**2 + sy1**2)
            
            sx2 = sobel(gray2, axis=1)
            sy2 = sobel(gray2, axis=0)
            edge2 = np.sqrt(sx2**2 + sy2**2)
            
            # 正規化
            max1 = np.max(edge1) + 1e-6
            max2 = np.max(edge2) + 1e-6
            edge1 = np.clip(edge1 / max1, 0, 1)
            edge2 = np.clip(edge2 / max2, 0, 1)
            
            # エッジマスク（どちらかのフレームにエッジがあれば True）
            edge_mask = np.maximum(edge1, edge2) > 0.1
            
            return edge_mask.astype(np.float32)
        except Exception as e:
            print(f"⚠ Edge detection failed: {e}")
            # フォールバック：エッジなしマスク
            return np.zeros(frame1.shape[:2], dtype=np.float32)
    
    def _apply_edge_preservation(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        blended: np.ndarray,
        edge_mask: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        エッジ領域を保存（エッジ周辺は元フレームを優先）
        """
        try:
            from scipy.ndimage import binary_dilation
            
            # エッジ領域の境界拡張
            dilated_mask = binary_dilation(edge_mask > 0.5, iterations=2).astype(np.float32)
            
            # エッジ領域では元フレームを使用
            frame1_edge = np.clip(frame1.astype(np.float32), 0, 1)
            frame2_edge = np.clip(frame2.astype(np.float32), 0, 1)
            
            edge_interpolated = (1 - t) * frame1_edge + t * frame2_edge
            
            # ブレンド
            mask_3d = dilated_mask[..., np.newaxis]
            result = blended * (1 - mask_3d) + edge_interpolated * mask_3d
            
            return result
        except Exception as e:
            print(f"⚠ Edge preservation failed: {e}")
            return blended
    
    @staticmethod
    def _apply_color_correction(
        frame1: np.ndarray,
        frame2: np.ndarray,
        blended: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        色域補正
        HSV 色空間で補間して RGB に変換
        """
        try:
            from skimage.color import rgb2hsv, hsv2rgb
            
            # 確実に 0-1 の float32 に正規化
            f1 = np.clip(frame1.astype(np.float32), 0, 1)
            f2 = np.clip(frame2.astype(np.float32), 0, 1)
            
            # RGB -> HSV
            hsv1 = rgb2hsv(f1)
            hsv2 = rgb2hsv(f2)
            
            # HSV で補間（Hue は円形なので特別な処理が必要）
            h1, s1, v1 = hsv1[..., 0], hsv1[..., 1], hsv1[..., 2]
            h2, s2, v2 = hsv2[..., 0], hsv2[..., 1], hsv2[..., 2]
            
            # Hue の円形補間
            h_diff = h2 - h1
            h_diff = np.where(h_diff > 0.5, h_diff - 1, h_diff)
            h_diff = np.where(h_diff < -0.5, h_diff + 1, h_diff)
            h_interp = h1 + t * h_diff
            h_interp = np.mod(h_interp, 1.0)
            
            # S, V は線形補間
            s_interp = (1 - t) * s1 + t * s2
            v_interp = (1 - t) * v1 + t * v2
            
            # HSV を RGB に変換
            hsv_interp = np.stack([h_interp, s_interp, v_interp], axis=-1)
            result = hsv2rgb(hsv_interp)
            
            return result
        except Exception as e:
            print(f"⚠ Color correction failed: {e}, using blended directly")
            return blended
    
    def _apply_animation_smoothing(self, frame: np.ndarray) -> np.ndarray:
        """
        アニメーション対応の平滑化
        バイラテラルフィルタで平滑化しながらエッジを保存
        """
        # ガウスフィルタ + メディアンフィルタの組み合わせ
        result = np.zeros_like(frame)
        
        for c in range(frame.shape[2] if len(frame.shape) == 3 else 1):
            if len(frame.shape) == 3:
                channel = frame[..., c]
            else:
                channel = frame
            
            # ガウスフィルタ
            smoothed = gaussian_filter(channel, sigma=1.0)
            
            # メディアンフィルタ（エッジ保存）
            median = median_filter(smoothed, size=3)
            
            # ブレンド
            filtered = 0.7 * smoothed + 0.3 * median
            
            if len(frame.shape) == 3:
                result[..., c] = filtered
            else:
                result = filtered
        
        return result
    
    def _bilateral_filter_channel(self, channel: np.ndarray, sigma_spatial: float = 1.5, sigma_color: float = 0.1) -> np.ndarray:
        """
        高速なエッジ保存フィルタ
        ガウスフィルタとメディアンフィルタの組み合わせ
        """
        # ガウスフィルタ
        smoothed = gaussian_filter(channel, sigma=sigma_spatial)
        
        # メディアンフィルタ（ノイズ除去、エッジ保存）
        median = median_filter(smoothed, size=3)
        
        # ブレンド
        result = 0.7 * smoothed + 0.3 * median
        
        return result
    
    def interpolate_with_edge_linking(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int
    ) -> List[np.ndarray]:
        """
        エッジリンク付き補間（より高度なアニメーション対応）
        
        特徴点のマッチングを使用してより正確な補間
        """
        try:
            # 特徴点検出と対応
            keypoints1 = self._extract_keypoints(frame1)
            keypoints2 = self._extract_keypoints(frame2)
            
            if keypoints1 is not None and keypoints2 is not None and len(keypoints1) >= 3 and len(keypoints2) >= 3:
                # 特徴点ベースの補間
                return self._interpolate_with_features(
                    frame1, frame2, keypoints1, keypoints2, num_frames
                )
        except Exception as e:
            print(f"⚠ Edge linking failed: {e}")
        
        # フォールバック：通常の補間
        return self.interpolate(frame1, frame2, num_frames, preserve_edges=True, use_color_correction=True)
    
    @staticmethod
    def _extract_keypoints(frame: np.ndarray) -> np.ndarray:
        """
        Harris コーナー検出で特徴点を抽出
        """
        from skimage.feature import corner_harris, corner_peaks
        
        # グレースケール変換
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        
        # コーナー検出
        corners = corner_harris(gray)
        peaks = corner_peaks(corners, min_distance=10, threshold_rel=0.05)
        
        if len(peaks) < 3:
            return None
        
        return peaks.astype(np.float32)
    
    def _interpolate_with_features(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        num_frames: int
    ) -> List[np.ndarray]:
        """
        特徴点ベースの補間
        """
        from scipy.spatial import Delaunay
        
        # 特徴点数を統一（少ない方に合わせる）
        min_points = min(len(keypoints1), len(keypoints2))
        keypoints1 = keypoints1[:min_points]
        keypoints2 = keypoints2[:min_points]
        
        if min_points < 3:
            # 特徴点が不足している場合は基本補間にフォールバック
            return self.interpolate(frame1, frame2, num_frames, preserve_edges=True)
        
        h, w = frame1.shape[:2]
        interpolated_frames = []
        
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)
            
            # 中間フレームの特徴点
            keypoints_mid = (1 - t) * keypoints1 + t * keypoints2
            
            # Delaunay 分割
            try:
                boundary = np.array([
                    [0, 0], [w, 0], [w, h], [0, h],
                    [w/2, 0], [w, h/2], [w/2, h], [0, h/2]
                ])
                all_points = np.vstack([keypoints_mid, boundary])
                delaunay = Delaunay(all_points)
                
                # フレーム補間
                frame = self._warp_with_delaunay(
                    frame1, frame2, keypoints1, keypoints2,
                    keypoints_mid, delaunay, h, w, t
                )
            except:
                # フォールバック
                frame = ((1 - t) * frame1 + t * frame2).astype(np.uint8)
            
            # アニメーション平滑化
            frame_float = frame.astype(np.float32) / 255.0
            frame_smooth = self._apply_animation_smoothing(frame_float)
            frame = np.clip(frame_smooth * 255, 0, 255).astype(np.uint8)
            
            interpolated_frames.append(frame)
        
        return interpolated_frames
    
    @staticmethod
    def _warp_with_delaunay(
        frame1: np.ndarray,
        frame2: np.ndarray,
        kp1: np.ndarray,
        kp2: np.ndarray,
        kp_mid: np.ndarray,
        delaunay,
        h: int,
        w: int,
        t: float
    ) -> np.ndarray:
        """
        Delaunay 分割を使用した局所ワーピング
        """
        from scipy.ndimage import map_coordinates
        
        result = np.zeros_like(frame1, dtype=np.float32)
        
        for simplex in delaunay.simplices:
            # 三角形領域の処理
            pts_mid = np.vstack([
                kp_mid[i] if i < len(kp_mid) else [0, 0]
                for i in simplex
            ])
            
            # 三角形の bounding box
            x_min = max(0, int(np.min(pts_mid[:, 0])))
            x_max = min(w, int(np.max(pts_mid[:, 0])) + 1)
            y_min = max(0, int(np.min(pts_mid[:, 1])))
            y_max = min(h, int(np.max(pts_mid[:, 1])) + 1)
            
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # アフィン座標計算
                    try:
                        A = pts_mid[:2, :2].T
                        b = np.array([x, y]) - pts_mid[0, :2]
                        bary = np.linalg.solve(A, b)
                        
                        if np.all(bary >= -0.01) and np.all(bary <= 1.01):
                            bary = np.clip(bary, 0, 1)
                            bary = np.append(bary, 1 - np.sum(bary))
                            
                            # ブレンド
                            result[y, x] = (1 - t) * frame1[y, x] + t * frame2[y, x]
                    except:
                        pass
        
        return result.astype(np.uint8)

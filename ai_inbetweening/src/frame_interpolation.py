"""
フレーム補間エンジンモジュール
簡易版のフレーム補間を実装
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List


class FrameInterpolator:
    """フレーム補間を行うクラス"""
    
    def __init__(self, model_type: str = 'rife', device: str = 'cpu'):
        """
        Initialize the frame interpolator
        
        Args:
            model_type: 使用するモデルのタイプ ('rife', 'morph', 'toon')
            device: PyTorchデバイス ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.device = device
        
        # ToonComposer スタイルの補間エンジンを初期化
        if model_type == 'toon':
            from .toon_style_interpolator import ToonStyleInterpolator
            self.toon_interpolator = ToonStyleInterpolator()
        else:
            self.toon_interpolator = None
        
        print(f"Frame Interpolator initialized with model: {model_type}")
    
    def interpolate(
        self, 
        frame1: np.ndarray, 
        frame2: np.ndarray, 
        num_frames: int = 5
    ) -> List[np.ndarray]:
        """
        2つのフレーム間に中割フレームを生成
        
        Args:
            frame1: 最初のフレーム (H, W, 3) uint8
            frame2: 2番目のフレーム (H, W, 3) uint8
            num_frames: 生成する中割フレーム数
        
        Returns:
            生成された中割フレームのリスト
        """
        # フレームサイズを揃える
        frame1, frame2 = self._align_frames(frame1, frame2)
        
        if self.model_type == 'toon':
            # ToonComposer スタイルの補間
            return self.toon_interpolator.interpolate_with_edge_linking(
                frame1, frame2, num_frames
            )
        elif self.model_type == 'dynamicrafter':
            # DynamiCrafter 統合（外部 SDK / API を利用するラッパー）
            try:
                from .dynamicrafter_integration import generate_inbetweens
                print("✓ Using DynamiCrafter integration")
                return generate_inbetweens(frame1, frame2, num_frames=num_frames)
            except Exception as e:
                print(f"⚠ DynamiCrafter integration failed: {e}")
                print("  Falling back to linear interpolation")
                return self._interpolate_linear(frame1, frame2, num_frames)
        elif self.model_type == 'rife':
            return self._interpolate_rife(frame1, frame2, num_frames)
        elif self.model_type == 'morph':
            return self.interpolate_with_morphing(frame1, frame2, num_frames)
        else:
            # デフォルト：線形補間
            return self._interpolate_linear(frame1, frame2, num_frames)
    
    @staticmethod
    def _align_frames(frame1: np.ndarray, frame2: np.ndarray) -> tuple:
        """
        2つのフレームサイズを揃える
        
        Args:
            frame1: 最初のフレーム
            frame2: 2番目のフレーム
        
        Returns:
            揃えられたフレームのタプル
        """
        from PIL import Image
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        if (h1, w1) == (h2, w2):
            return frame1, frame2
        
        # 小さい方のサイズに揃える
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        print(f"⚠ Frame size mismatch: ({h1}x{w1}) vs ({h2}x{w2})")
        print(f"  Resizing to: {target_h}x{target_w}")
        
        # PIL を使用してリサイズ
        img1 = Image.fromarray(frame1).resize((target_w, target_h), Image.LANCZOS)
        img2 = Image.fromarray(frame2).resize((target_w, target_h), Image.LANCZOS)
        
        frame1_resized = np.array(img1)
        frame2_resized = np.array(img2)
        
        return frame1_resized, frame2_resized
    
    def _interpolate_linear(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int
    ) -> List[np.ndarray]:
        """
        線形補間によるフレーム生成（簡易版）
        
        Args:
            frame1: 最初のフレーム
            frame2: 2番目のフレーム
            num_frames: 生成するフレーム数
        
        Returns:
            補間されたフレームのリスト
        """
        frame1_float = frame1.astype(np.float32) / 255.0
        frame2_float = frame2.astype(np.float32) / 255.0
        
        interpolated_frames = []
        
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)  # 0 < t < 1
            
            # 線形補間
            interpolated = (1 - t) * frame1_float + t * frame2_float
            interpolated_uint8 = (interpolated * 255).astype(np.uint8)
            
            interpolated_frames.append(interpolated_uint8)
        
        return interpolated_frames
    
    def _interpolate_rife(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int
    ) -> List[np.ndarray]:
        """
        RIFE ベースのフレーム補間 (PyTorch ベース実装)
        
        実装: 光学フロー + ワーピングを使用した自然な中間フレーム生成
        """
        try:
            import torch
            import torch.nn.functional as F
            
            # RIFE モデルの実装（簡易版）
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🎬 RIFE interpolation using {device}")
            
            interpolated_frames = []
            
            # フレームを torch.Tensor に変換
            frame1_tensor = self._numpy_to_tensor(frame1, device)  # [1, 3, H, W]
            frame2_tensor = self._numpy_to_tensor(frame2, device)  # [1, 3, H, W]
            
            # デバッグ: フレームの統計情報を出力
            print(f"Frame1 stats: min={frame1.min()}, max={frame1.max()}, mean={frame1.mean()}")
            print(f"Frame2 stats: min={frame2.min()}, max={frame2.max()}, mean={frame2.mean()}")

            # 複数フレーム生成
            for i in range(1, num_frames + 1):
                t = i / (num_frames + 1)
                
                # 光学フロー + ワーピング による補間
                intermediate_tensor = self._interpolate_with_flow(
                    frame1_tensor, frame2_tensor, t
                )
                
                # Tensor を NumPy に変換
                intermediate = self._tensor_to_numpy(intermediate_tensor)
                interpolated_frames.append(intermediate)
            
            # フレーム補間の結果を確認
            for i, frame in enumerate(interpolated_frames):
                print(f"Interpolated frame {i} stats: min={frame.min()}, max={frame.max()}, mean={frame.mean()}")

            return interpolated_frames
            
        except Exception as e:
            print(f"⚠ RIFE interpolation failed: {e}")
            print("  Falling back to linear interpolation")
            return self._interpolate_linear(frame1, frame2, num_frames)
    
    def _numpy_to_tensor(
        self,
        frame: np.ndarray,
        device: str
    ) -> "torch.Tensor":
        """NumPy 配列を PyTorch Tensor に変換"""
        import torch
        
        # フレームを float32 に正規化 (0-1)
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0
        
        # NumPy [H, W, C] → PyTorch [1, C, H, W]
        if frame.shape[2] == 3:  # RGB
            frame = np.transpose(frame, (2, 0, 1))  # [C, H, W]
        
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).to(device)  # [1, C, H, W]
        return frame_tensor
    
    def _tensor_to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        """PyTorch Tensor を NumPy 配列に変換"""
        # Tensor [1, C, H, W] → NumPy [H, W, C]
        tensor = tensor.squeeze(0).cpu().detach()  # [C, H, W]
        frame = torch.clamp(tensor, 0, 1).permute(1, 2, 0).numpy()  # [H, W, C]
        frame = (frame * 255).astype(np.uint8)
        return frame
    
    def _interpolate_with_flow(
        self,
        frame1: "torch.Tensor",
        frame2: "torch.Tensor",
        t: float
    ) -> "torch.Tensor":
        """
        光学フロー + ワーピングによる補間
        ポーズ変化とスケール変化に対応した高度な補間
        """
        import torch
        import torch.nn.functional as F
        
        # Tensor を NumPy に変換して光学フロー計算
        frame1_np = frame1.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        frame2_np = frame2.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        
        # 光学フロー計算
        try:
            flow = self._compute_optical_flow(frame1_np, frame2_np)
            
            # フロー情報を使用したワーピング
            warped_frame1 = self._warp_frame_with_flow(frame1_np, flow, 1.0 - t)
            warped_frame2 = self._warp_frame_with_flow(frame2_np, flow, -t)
            
            # ワープされたフレームをブレンド
            blended = (1.0 - t) * warped_frame1 + t * warped_frame2
            
        except Exception as e:
            # フロー計算失敗時は簡易補間
            blended = (1.0 - t) * frame1_np + t * frame2_np
        
        # NumPy を Tensor に変換
        blended_tensor = torch.from_numpy(blended).permute(2, 0, 1).unsqueeze(0).to(frame1.device)
        blended_tensor = torch.clamp(blended_tensor, 0, 1)
        
        return blended_tensor
    
    @staticmethod
    def _compute_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Dense optical flow を計算
        scikit-image を使用
        
        Returns:
            フロー配列 [H, W, 2]
        """
        try:
            # OpenCV が利用可能かチェック（GUI 機能なしで）
            from skimage.feature import match_template
            from scipy import signal
            
            # グレースケール変換
            gray1 = np.dot(frame1[..., :3], [0.299, 0.587, 0.114])
            gray2 = np.dot(frame2[..., :3], [0.299, 0.587, 0.114])
            
            # 簡易的な光学フロー計算（勾配ベース）
            # Sobel フィルタで時間勾配を計算
            from scipy.ndimage import sobel
            
            # 空間勾配
            gx = sobel(gray1, axis=1)
            gy = sobel(gray1, axis=0)
            
            # 時間勾配
            gt = gray2.astype(np.float32) - gray1.astype(np.float32)
            
            # Lucas-Kanade アルゴリズム（簡略版）
            # ウィンドウサイズ
            win_size = 15
            h, w = gray1.shape
            
            flow = np.zeros((h, w, 2), dtype=np.float32)
            
            for y in range(win_size, h - win_size, win_size // 2):
                for x in range(win_size, w - win_size, win_size // 2):
                    # ウィンドウ抽出
                    window = slice(y - win_size // 2, y + win_size // 2 + 1)
                    window_x = slice(x - win_size // 2, x + win_size // 2 + 1)
                    
                    gx_win = gx[window, window_x].flatten()
                    gy_win = gy[window, window_x].flatten()
                    gt_win = gt[window, window_x].flatten()
                    
                    # 最小二乗法で解く
                    A = np.vstack([gx_win, gy_win]).T
                    b = -gt_win
                    
                    try:
                        flow_win, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        flow[window, window_x] = flow_win
                    except:
                        pass
            
            # フロー平滑化
            from scipy.ndimage import gaussian_filter
            flow[:, :, 0] = gaussian_filter(flow[:, :, 0], sigma=2)
            flow[:, :, 1] = gaussian_filter(flow[:, :, 1], sigma=2)
            
            return flow
            
        except Exception as e:
            print(f"⚠ Optical flow computation failed: {e}")
            # ゼロフローをリターン
            return np.zeros((*frame1.shape[:2], 2), dtype=np.float32)
    
    @staticmethod
    def _warp_frame_with_flow(frame: np.ndarray, flow: np.ndarray, factor: float) -> np.ndarray:
        """
        フロー情報を使用してフレームをワープ
        scipy.ndimage を使用した実装
        
        Args:
            frame: 入力フレーム [H, W, 3] (0-1 float)
            flow: 光学フロー [H, W, 2]
            factor: ワープの強度 (0-1)
        
        Returns:
            ワープされたフレーム
        """
        from scipy.ndimage import map_coordinates
        
        h, w = frame.shape[:2]
        
        # メッシュグリッドを作成
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # フロー適用
        map_x = (x + flow[..., 0] * factor).astype(np.float32)
        map_y = (y + flow[..., 1] * factor).astype(np.float32)
        
        # 範囲外の値をクリップ
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        # スタック座標
        coords = np.array([map_y, map_x, np.array([0, 1, 2])])
        
        # ワープ処理
        warped = np.zeros_like(frame)
        
        try:
            for c in range(frame.shape[2]):
                channel_coords = np.array([map_y, map_x])
                warped[:, :, c] = map_coordinates(
                    (frame[:, :, c] * 255).astype(np.uint8),
                    channel_coords,
                    order=1,
                    mode='reflect'
                )
        except:
            # 処理失敗時は元フレームを返す
            warped = frame
        
        return warped.astype(np.float32) / 255.0
    
    def interpolate_with_timing(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int,
        easing: str = 'linear'
    ) -> List[np.ndarray]:
        """
        イージング関数を使用したフレーム補間
        
        Args:
            frame1: 最初のフレーム
            frame2: 2番目のフレーム
            num_frames: 生成するフレーム数
            easing: イージングタイプ ('linear', 'ease_in', 'ease_out', 'ease_in_out')
        
        Returns:
            補間されたフレームのリスト
        """
        frame1_float = frame1.astype(np.float32) / 255.0
        frame2_float = frame2.astype(np.float32) / 255.0
        
        interpolated_frames = []
        
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)
            
            # イージング関数を適用
            t_eased = self._apply_easing(t, easing)
            
            interpolated = (1 - t_eased) * frame1_float + t_eased * frame2_float
            interpolated_uint8 = (interpolated * 255).astype(np.uint8)
            
            interpolated_frames.append(interpolated_uint8)
        
        return interpolated_frames
    
    def interpolate_with_morphing(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int,
        use_feature_matching: bool = True
    ) -> List[np.ndarray]:
        """
        高度なモーフィング機能付きフレーム補間
        ポーズ変化とスケール変化を正確に補間
        
        Args:
            frame1: 最初のフレーム
            frame2: 2番目のフレーム
            num_frames: 生成するフレーム数
            use_feature_matching: 特徴点マッチング使用フラグ
        
        Returns:
            補間されたフレームのリスト
        """
        interpolated_frames = []
        
        try:
            if use_feature_matching:
                # 特徴点ベースのモーフィング
                keypoints1, keypoints2 = self._match_features(frame1, frame2)
                
                if keypoints1 is not None and len(keypoints1) >= 3:
                    for i in range(1, num_frames + 1):
                        t = i / (num_frames + 1)
                        morphed = self._morph_with_keypoints(
                            frame1, frame2, keypoints1, keypoints2, t
                        )
                        interpolated_frames.append(morphed)
                    
                    print(f"✅ Morphing with {len(keypoints1)} keypoints")
                    return interpolated_frames
        
        except Exception as e:
            print(f"⚠ Feature matching failed: {e}")
        
        # フォールバック: 光学フロー補間
        return self._interpolate_rife(frame1, frame2, num_frames)
    
    @staticmethod
    def _match_features(frame1: np.ndarray, frame2: np.ndarray) -> tuple:
        """
        2フレーム間の特徴点をマッチング
        scikit-image を使用した実装
        
        Returns:
            (keypoints1, keypoints2): マッチされた特徴点のペア
        """
        from skimage.feature import corner_peaks, corner_harris
        from skimage.measure import ransac
        from skimage.transform import EuclideanTransform
        
        # グレースケール変換
        gray1 = np.dot(frame1[..., :3], [0.299, 0.587, 0.114])
        gray2 = np.dot(frame2[..., :3], [0.299, 0.587, 0.114])
        
        # コーナー検出
        corners1 = corner_harris(gray1)
        corners2 = corner_harris(gray2)
        
        # ピークの検出
        peaks1 = corner_peaks(corners1, min_distance=5, threshold_rel=0.1)
        peaks2 = corner_peaks(corners2, min_distance=5, threshold_rel=0.1)
        
        if len(peaks1) < 3 or len(peaks2) < 3:
            return None, None
        
        # 簡易的な特徴マッチング（最近傍探索）
        from scipy.spatial.distance import cdist
        
        # コーナー周辺のパッチを抽出
        patch_size = 10
        matches = []
        
        for i, p1 in enumerate(peaks1[:50]):
            y1, x1 = p1
            
            # パッチ抽出
            y_start = max(0, y1 - patch_size // 2)
            y_end = min(gray1.shape[0], y1 + patch_size // 2 + 1)
            x_start = max(0, x1 - patch_size // 2)
            x_end = min(gray1.shape[1], x1 + patch_size // 2 + 1)
            
            patch1 = gray1[y_start:y_end, x_start:x_end].flatten()
            
            best_match = None
            best_dist = float('inf')
            
            for j, p2 in enumerate(peaks2[:50]):
                y2, x2 = p2
                
                # パッチ抽出
                y_start2 = max(0, y2 - patch_size // 2)
                y_end2 = min(gray2.shape[0], y2 + patch_size // 2 + 1)
                x_start2 = max(0, x2 - patch_size // 2)
                x_end2 = min(gray2.shape[1], x2 + patch_size // 2 + 1)
                
                patch2 = gray2[y_start2:y_end2, x_start2:x_end2].flatten()
                
                # パッチサイズが異なる場合はスキップ
                if len(patch1) != len(patch2):
                    continue
                
                # ユークリッド距離を計算
                dist = np.sum((patch1 - patch2) ** 2) ** 0.5
                
                if dist < best_dist:
                    best_dist = dist
                    best_match = j
            
            if best_match is not None and best_dist < 500:
                matches.append((i, best_match, best_dist))
        
        if len(matches) < 3:
            return None, None
        
        # マッチを距離でソート
        matches = sorted(matches, key=lambda m: m[2])[:30]
        
        # マッチ点を抽出
        keypoints1 = np.float32([peaks1[m[0]] for m in matches])
        keypoints2 = np.float32([peaks2[m[1]] for m in matches])
        
        return keypoints1, keypoints2
    
    @staticmethod
    def _morph_with_keypoints(
        frame1: np.ndarray,
        frame2: np.ndarray,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        特徴点に基づいてフレームをモーフィング
        scipy を使用した実装
        
        Args:
            frame1, frame2: 入力フレーム
            keypoints1, keypoints2: マッチされた特徴点
            t: 補間パラメータ (0-1)
        
        Returns:
            モーフィングされたフレーム
        """
        from scipy.spatial import Delaunay
        from scipy.ndimage import map_coordinates
        
        h, w = frame1.shape[:2]
        
        # 中間フレームの特徴点を計算
        keypoints_mid = (1 - t) * keypoints1 + t * keypoints2
        
        # Delaunay 三角形分割
        try:
            # 画像境界点を追加
            boundary_pts = np.array([
                [0, 0], [w-1, 0], [w-1, h-1], [0, h-1],
                [w//2, 0], [w-1, h//2], [w//2, h-1], [0, h//2]
            ])
            
            all_points = np.vstack([keypoints_mid, boundary_pts])
            delaunay = Delaunay(all_points)
            
        except Exception as e:
            print(f"⚠ Delaunay triangulation failed: {e}")
            # フォールバック：簡易補間
            return ((1.0 - t) * frame1 + t * frame2).astype(np.uint8)
        
        # フレームをモーフィング
        morphed = np.zeros_like(frame1, dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        for simplex in delaunay.simplices:
            # 三角形の頂点を抽出
            pts_mid = all_points[simplex]
            pts1 = np.vstack([keypoints1[i] if i < len(keypoints1) else all_points[i] 
                            for i in simplex])
            pts2 = np.vstack([keypoints2[i] if i < len(keypoints2) else all_points[i] 
                            for i in simplex])
            
            # 三角形の領域を取得
            x_min, x_max = int(max(0, np.min(pts_mid[:, 0]))), int(min(w-1, np.max(pts_mid[:, 0])))
            y_min, y_max = int(max(0, np.min(pts_mid[:, 1]))), int(min(h-1, np.max(pts_mid[:, 1])))
            
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    # 重心座標を計算
                    pt = np.array([x, y])
                    
                    try:
                        # 逆アフィン変換
                        A = np.vstack([pts_mid.T, [1, 1, 1]])
                        b = np.hstack([pt, 1])
                        bary = np.linalg.solve(A[:, :2].T, (pt - pts_mid[0]))
                        
                        if np.all(bary >= -0.01) and np.all(bary <= 1.01):
                            # 点が三角形内にある
                            bary = np.clip(bary, 0, 1)
                            bary = np.append(bary, 1 - np.sum(bary))
                            
                            # フレーム 1 と 2 の対応する点を計算
                            pt1 = np.dot(bary[:len(pts1)], pts1)
                            pt2 = np.dot(bary[:len(pts2)], pts2)
                            
                            # 補間
                            pt_interp = (1 - t) * pt1 + t * pt2
                            
                            # バイリニア補間で色を取得
                            px, py = pt_interp
                            px = np.clip(px, 0, w - 1)
                            py = np.clip(py, 0, h - 1)
                            
                            px_int, py_int = int(px), int(py)
                            px_frac = px - px_int
                            py_frac = py - py_int
                            
                            # バイリニア補間
                            c1 = frame1[py_int, px_int] * (1 - px_frac) * (1 - py_frac)
                            c2 = frame1[py_int, min(px_int + 1, w-1)] * px_frac * (1 - py_frac)
                            c3 = frame1[min(py_int + 1, h-1), px_int] * (1 - px_frac) * py_frac
                            c4 = frame1[min(py_int + 1, h-1), min(px_int + 1, w-1)] * px_frac * py_frac
                            
                            color1 = c1 + c2 + c3 + c4
                            
                            # フレーム 2 からも同様に取得
                            c1 = frame2[py_int, px_int] * (1 - px_frac) * (1 - py_frac)
                            c2 = frame2[py_int, min(px_int + 1, w-1)] * px_frac * (1 - py_frac)
                            c3 = frame2[min(py_int + 1, h-1), px_int] * (1 - px_frac) * py_frac
                            c4 = frame2[min(py_int + 1, h-1), min(px_int + 1, w-1)] * px_frac * py_frac
                            
                            color2 = c1 + c2 + c3 + c4
                            
                            # ブレンド
                            morphed[y, x] = (1 - t) * color1 + t * color2
                            count[y, x] += 1
                    except:
                        pass
        
        # 未処理領域を簡易補間で埋める
        mask = count == 0
        morphed[mask] = ((1.0 - t) * frame1[mask] + t * frame2[mask])
        
        # 正規化
        morphed = np.clip(morphed, 0, 255).astype(np.uint8)
        
        return morphed
    
    @staticmethod
    def _apply_easing(t: float, easing: str) -> float:
        """
        イージング関数を適用
        
        Args:
            t: 0-1の値
            easing: イージングタイプ
        
        Returns:
            イージング適用後の値
        """
        if easing == 'linear':
            return t
        elif easing == 'ease_in':
            return t * t
        elif easing == 'ease_out':
            return t * (2 - t)
        elif easing == 'ease_in_out':
            if t < 0.5:
                return 2 * t * t
            else:
                return -1 + (4 - 2 * t) * t
        else:
            return t

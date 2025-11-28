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
            model_type: 使用するモデルのタイプ
            device: PyTorchデバイス ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.device = device
        
        # TODO: 実際のモデルをここで読み込む
        # 現在は簡易版（線形補間）を使用
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
        
        if self.model_type == 'rife':
            return self._interpolate_rife(frame1, frame2, num_frames)
        else:
            # デフォルトは線形補間
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
        RIFE モデルを使用したフレーム補間
        
        TODO: 実際のRIFEモデルの実装
        """
        # 現在は線形補間を使用
        print("Note: RIFE model not yet implemented. Using linear interpolation.")
        return self._interpolate_linear(frame1, frame2, num_frames)
    
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

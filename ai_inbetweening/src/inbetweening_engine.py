"""
AI Inbetweening Engine - Main module
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Union
from .keyframe_loader import KeyframeLoader
from .frame_interpolation import FrameInterpolator


class InbetWeeningEngine:
    """キーフレーム画像から中割を生成するメインエンジン"""
    
    def __init__(self, device: str = None, model_type: str = 'rife'):
        """
        Initialize the inbetweening engine
        
        Args:
            device: 'cuda' or 'cpu' (自動判定)
            model_type: フレーム補間モデルのタイプ ('rife', 'dain', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        self.keyframe_loader = KeyframeLoader()
        self.interpolator = FrameInterpolator(model_type=model_type, device=self.device)
        
        print(f"InbetWeeningEngine initialized with device: {self.device}")
        print(f"Using model: {self.model_type}")
    
    def generate(
        self, 
        keyframe1_path: Union[str, Path], 
        keyframe2_path: Union[str, Path], 
        num_frames: int = 5,
        save_path: Union[str, Path] = None
    ) -> List[np.ndarray]:
        """
        キーフレーム2枚の間に中割を生成
        
        Args:
            keyframe1_path: 最初のキーフレーム画像パス
            keyframe2_path: 2番目のキーフレーム画像パス
            num_frames: 生成する中割フレーム数 (デフォルト: 5)
            save_path: 出力先ディレクトリ (Noneの場合は保存しない)
        
        Returns:
            生成されたフレームのNumPy配列リスト
        """
        # キーフレーム画像を読み込む
        frame1 = self.keyframe_loader.load(keyframe1_path)
        frame2 = self.keyframe_loader.load(keyframe2_path)
        
        print(f"Loaded keyframes: {frame1.shape}, {frame2.shape}")
        
        # フレーム補間を実行
        interpolated_frames = self.interpolator.interpolate(
            frame1, frame2, num_frames=num_frames
        )
        
        # 全フレーム（キーフレーム1 + 補間フレーム + キーフレーム2）
        all_frames = [frame1] + interpolated_frames + [frame2]
        
        if save_path:
            self._save_frames(all_frames, save_path)
        
        return all_frames
    
    def generate_sequence(
        self,
        keyframes_paths: List[Union[str, Path]],
        num_frames: int = 5,
        save_path: Union[str, Path] = None
    ) -> List[np.ndarray]:
        """
        複数のキーフレームから連続したアニメーションを生成
        
        Args:
            keyframes_paths: キーフレーム画像パスのリスト
            num_frames: 各区間の中割フレーム数
            save_path: 出力先ディレクトリ
        
        Returns:
            生成されたすべてのフレームのリスト
        """
        all_frames = []
        
        for i in range(len(keyframes_paths) - 1):
            frames = self.generate(
                keyframes_paths[i], 
                keyframes_paths[i + 1], 
                num_frames=num_frames,
                save_path=None
            )
            # 最初のフレーム以外を追加（重複を避ける）
            all_frames.extend(frames[1:])
        
        if save_path:
            self._save_frames(all_frames, save_path)
        
        return all_frames
    
    def _save_frames(self, frames: List[np.ndarray], save_path: Union[str, Path]) -> None:
        """フレームをファイルに保存"""
        from PIL import Image
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for idx, frame in enumerate(frames):
            # NumPy配列をPIL Imageに変換
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            
            img = Image.fromarray(frame)
            output_file = save_path / f"frame_{idx:04d}.png"
            img.save(output_file)
            print(f"Saved: {output_file}")
    
    def export_video(
        self, 
        frames: List[np.ndarray], 
        output_path: Union[str, Path],
        fps: int = 24
    ) -> None:
        """フレームを動画ファイルにエクスポート"""
        import imageio_ffmpeg
        import imageio
        
        output_path = str(output_path)
        
        # NumPy配列をuint8に変換
        frames_uint8 = []
        for frame in frames:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            frames_uint8.append(frame)
        
        # imageio-ffmpeg を使用して MP4 を生成
        try:
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
            for frame in frames_uint8:
                writer.append_data(frame)
            writer.close()
        except Exception as e:
            # フォールバック: OpenCV を使用（インストールされている場合）
            print(f"⚠ FFmpeg write failed: {e}, trying alternative method...")
            try:
                import cv2
                h, w = frames_uint8[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                for frame in frames_uint8:
                    if len(frame.shape) == 2:  # グレースケール
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)
                out.release()
            except Exception as e2:
                print(f"✗ Both methods failed: {e2}")
                raise
        
        print(f"Video saved: {output_path}")

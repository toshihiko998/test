"""
キーフレーム画像ローダーモジュール
"""

from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union


class KeyframeLoader:
    """キーフレーム画像を読み込むためのクラス"""
    
    def __init__(self, target_size: tuple = None):
        """
        Initialize the keyframe loader
        
        Args:
            target_size: 読み込み時にリサイズするサイズ (width, height)
                         Noneの場合は元のサイズを保持
        """
        self.target_size = target_size
    
    def load(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        キーフレーム画像を読み込む
        
        Args:
            image_path: 画像ファイルのパス
        
        Returns:
            RGB形式のNumPy配列 (H, W, 3)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Pillow で画像を読み込む（RGB形式）
        img = Image.open(str(image_path)).convert('RGB')
        
        # リサイズ処理
        if self.target_size is not None:
            img = img.resize(self.target_size, Image.LANCZOS)
        
        # NumPy配列に変換
        image_rgb = np.array(img)
        
        return image_rgb.astype(np.uint8)
    
    def load_batch(self, image_paths: list) -> list:
        """
        複数のキーフレーム画像を一括読み込み
        
        Args:
            image_paths: 画像ファイルパスのリスト
        
        Returns:
            読み込まれたNumPy配列のリスト
        """
        frames = []
        for path in image_paths:
            frame = self.load(path)
            frames.append(frame)
        
        return frames
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        画像を正規化 (0-1の範囲に)
        
        Args:
            image: 0-255のNumPy配列
        
        Returns:
            0-1に正規化されたNumPy配列 (float32)
        """
        return image.astype(np.float32) / 255.0
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        正規化された画像を逆正規化 (0-255に)
        
        Args:
            image: 0-1のNumPy配列
        
        Returns:
            0-255のNumPy配列 (uint8)
        """
        return (image * 255).astype(np.uint8)

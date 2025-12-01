"""
KeyframeLoader ユニットテスト
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from src.keyframe_loader import KeyframeLoader


class TestKeyframeLoader:
    """KeyframeLoader のテストクラス"""
    
    @pytest.fixture
    def loader(self):
        """テスト用のLoaderインスタンスを作成"""
        return KeyframeLoader()
    
    @pytest.fixture
    def loader_with_resize(self):
        """リサイズ機能付きLoaderインスタンス"""
        return KeyframeLoader(target_size=(128, 128))
    
    @pytest.fixture
    def test_image_path(self, tmp_path):
        """テスト用の画像ファイルを作成"""
        img = Image.new('RGB', (256, 256), color='red')
        path = tmp_path / 'test_image.png'
        img.save(path)
        return path
    
    @pytest.fixture
    def test_rgba_image_path(self, tmp_path):
        """テスト用のRGBA画像ファイルを作成"""
        img = Image.new('RGBA', (256, 256), color=(255, 0, 0, 128))
        path = tmp_path / 'test_rgba_image.png'
        img.save(path)
        return path
    
    def test_load_returns_numpy_array(self, loader, test_image_path):
        """load()がNumPy配列を返すことを確認"""
        result = loader.load(test_image_path)
        assert isinstance(result, np.ndarray)
    
    def test_load_returns_correct_shape(self, loader, test_image_path):
        """load()が正しい形状の配列を返すことを確認"""
        result = loader.load(test_image_path)
        assert result.shape == (256, 256, 3)
    
    def test_load_returns_uint8(self, loader, test_image_path):
        """load()がuint8型の配列を返すことを確認"""
        result = loader.load(test_image_path)
        assert result.dtype == np.uint8
    
    def test_load_rgba_converts_to_rgb(self, loader, test_rgba_image_path):
        """RGBA画像がRGBに変換されることを確認"""
        result = loader.load(test_rgba_image_path)
        assert result.shape[2] == 3  # RGB, not RGBA
    
    def test_load_with_resize(self, loader_with_resize, test_image_path):
        """リサイズ機能が正しく動作することを確認"""
        result = loader_with_resize.load(test_image_path)
        assert result.shape == (128, 128, 3)
    
    def test_load_nonexistent_file(self, loader):
        """存在しないファイルを読み込もうとするとFileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            loader.load('/nonexistent/path/image.png')
    
    def test_load_batch(self, loader, tmp_path):
        """複数の画像を一括読み込みできることを確認"""
        paths = []
        for i in range(3):
            img = Image.new('RGB', (64, 64), color=(i * 80, 0, 0))
            path = tmp_path / f'batch_image_{i}.png'
            img.save(path)
            paths.append(path)
        
        results = loader.load_batch(paths)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (64, 64, 3)
    
    def test_normalize(self, loader):
        """normalize()が0-1の範囲に正規化することを確認"""
        image = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = loader.normalize(image)
        
        assert result.dtype == np.float32
        assert result[0, 0, 0] == 0.0
        assert np.isclose(result[0, 0, 1], 128/255)
        assert result[0, 0, 2] == 1.0
    
    def test_denormalize(self, loader):
        """denormalize()が0-255の範囲に戻すことを確認"""
        image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        result = loader.denormalize(image)
        
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 127 or result[0, 0, 1] == 128  # rounding
        assert result[0, 0, 2] == 255
    
    def test_normalize_denormalize_roundtrip(self, loader):
        """normalize -> denormalize で元の値に近い値が得られることを確認"""
        original = np.array([[[100, 150, 200]]], dtype=np.uint8)
        normalized = loader.normalize(original)
        denormalized = loader.denormalize(normalized)
        
        assert np.allclose(original, denormalized, atol=1)

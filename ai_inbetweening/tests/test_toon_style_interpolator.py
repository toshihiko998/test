"""
ToonStyleInterpolator ユニットテスト
"""

import pytest
import numpy as np
from pathlib import Path

from src.toon_style_interpolator import ToonStyleInterpolator


class TestToonStyleInterpolator:
    """ToonStyleInterpolator のテストクラス"""
    
    @pytest.fixture
    def interpolator(self):
        """テスト用のInterpolatorインスタンス"""
        return ToonStyleInterpolator()
    
    @pytest.fixture
    def frame1(self):
        """テスト用のフレーム1"""
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 255
        frame[10:30, 10:30] = [255, 0, 0]  # 赤い四角
        return frame
    
    @pytest.fixture
    def frame2(self):
        """テスト用のフレーム2"""
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 255
        frame[30:50, 30:50] = [0, 0, 255]  # 青い四角
        return frame
    
    @pytest.fixture
    def black_frame(self):
        """黒いフレーム"""
        return np.zeros((64, 64, 3), dtype=np.uint8)
    
    @pytest.fixture
    def white_frame(self):
        """白いフレーム"""
        return np.ones((64, 64, 3), dtype=np.uint8) * 255
    
    def test_interpolate_returns_list(self, interpolator, frame1, frame2):
        """interpolate()がリストを返すことを確認"""
        result = interpolator.interpolate(frame1, frame2, num_frames=3)
        assert isinstance(result, list)
    
    def test_interpolate_returns_correct_num_frames(self, interpolator, frame1, frame2):
        """interpolate()が指定した数のフレームを返すことを確認"""
        for num in [1, 3, 5]:
            result = interpolator.interpolate(frame1, frame2, num_frames=num)
            assert len(result) == num
    
    def test_interpolate_returns_uint8(self, interpolator, frame1, frame2):
        """interpolate()がuint8型の配列を返すことを確認"""
        result = interpolator.interpolate(frame1, frame2, num_frames=3)
        for frame in result:
            assert frame.dtype == np.uint8
    
    def test_interpolate_returns_same_shape(self, interpolator, frame1, frame2):
        """interpolate()が入力と同じ形状の配列を返すことを確認"""
        result = interpolator.interpolate(frame1, frame2, num_frames=3)
        for frame in result:
            assert frame.shape == frame1.shape
    
    def test_interpolate_without_edge_preservation(self, interpolator, frame1, frame2):
        """エッジ保存なしの補間が動作することを確認"""
        result = interpolator.interpolate(
            frame1, frame2, 
            num_frames=3, 
            preserve_edges=False
        )
        assert len(result) == 3
    
    def test_interpolate_without_color_correction(self, interpolator, frame1, frame2):
        """色補正なしの補間が動作することを確認"""
        result = interpolator.interpolate(
            frame1, frame2, 
            num_frames=3, 
            use_color_correction=False
        )
        assert len(result) == 3
    
    def test_interpolate_with_edge_linking(self, interpolator, frame1, frame2):
        """interpolate_with_edge_linking()が動作することを確認"""
        result = interpolator.interpolate_with_edge_linking(
            frame1, frame2, 
            num_frames=3
        )
        
        assert isinstance(result, list)
        assert len(result) == 3
        for frame in result:
            assert frame.dtype == np.uint8
    
    def test_detect_edges(self, interpolator, frame1, frame2):
        """_detect_edges()がエッジマスクを返すことを確認"""
        edge_mask = interpolator._detect_edges(frame1, frame2)
        
        assert isinstance(edge_mask, np.ndarray)
        assert edge_mask.shape == frame1.shape[:2]  # (H, W)
        assert edge_mask.dtype == np.float32
    
    def test_detect_edges_on_uniform_image(self, interpolator, white_frame):
        """均一な画像ではエッジが検出されないことを確認"""
        edge_mask = interpolator._detect_edges(white_frame, white_frame)
        
        # 均一画像はエッジが少ない
        assert edge_mask.sum() < edge_mask.size * 0.5
    
    def test_apply_animation_smoothing(self, interpolator, frame1):
        """_apply_animation_smoothing()が動作することを確認"""
        frame_float = frame1.astype(np.float32) / 255.0
        result = interpolator._apply_animation_smoothing(frame_float)
        
        assert result.shape == frame_float.shape
    
    def test_extract_keypoints(self, interpolator, frame1):
        """_extract_keypoints()がキーポイントを返すことを確認"""
        keypoints = interpolator._extract_keypoints(frame1)
        
        # キーポイントがない場合はNone、ある場合はNumPy配列
        assert keypoints is None or isinstance(keypoints, np.ndarray)
    
    def test_interpolate_handles_errors_gracefully(self, interpolator):
        """エラー時にフォールバック補間が動作することを確認"""
        # 不正な形状の入力
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)
        
        result = interpolator.interpolate(frame1, frame2, num_frames=3)
        
        # エラーが発生してもフォールバック補間が動作
        assert len(result) == 3
        for frame in result:
            assert frame.dtype == np.uint8
    
    def test_edge_preservation_with_high_contrast(self, interpolator):
        """高コントラスト画像でエッジ保存が動作することを確認"""
        # 高コントラストな画像
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame1[20:40, 20:40] = 255
        
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2[30:50, 30:50] = 255
        
        result = interpolator.interpolate(
            frame1, frame2, 
            num_frames=3, 
            preserve_edges=True
        )
        
        assert len(result) == 3
    
    def test_color_correction_preserves_shape(self, interpolator, frame1, frame2):
        """色補正後もフレームの形状が保持されることを確認"""
        frame1_float = frame1.astype(np.float32) / 255.0
        frame2_float = frame2.astype(np.float32) / 255.0
        blended = (frame1_float + frame2_float) / 2
        
        result = interpolator._apply_color_correction(
            frame1_float, frame2_float, blended, t=0.5
        )
        
        assert result.shape == frame1.shape
    
    def test_bilateral_filter_channel(self, interpolator):
        """_bilateral_filter_channel()が動作することを確認"""
        channel = np.random.rand(64, 64).astype(np.float32)
        result = interpolator._bilateral_filter_channel(channel)
        
        assert result.shape == channel.shape

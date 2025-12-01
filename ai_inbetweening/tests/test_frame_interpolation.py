"""
FrameInterpolator ユニットテスト
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.frame_interpolation import FrameInterpolator


class TestFrameInterpolator:
    """FrameInterpolator のテストクラス"""
    
    @pytest.fixture
    def interpolator(self):
        """テスト用のInterpolatorインスタンス（RIFEモード）"""
        return FrameInterpolator(model_type='rife', device='cpu')
    
    @pytest.fixture
    def linear_interpolator(self):
        """線形補間用のInterpolatorインスタンス"""
        return FrameInterpolator(model_type='linear', device='cpu')
    
    @pytest.fixture
    def morph_interpolator(self):
        """モーフィング補間用のInterpolatorインスタンス"""
        return FrameInterpolator(model_type='morph', device='cpu')
    
    @pytest.fixture
    def frame1(self):
        """テスト用のフレーム1（白い背景に赤い四角）"""
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 255
        frame[10:20, 10:20] = [255, 0, 0]  # 赤い四角
        return frame
    
    @pytest.fixture
    def frame2(self):
        """テスト用のフレーム2（白い背景に青い四角）"""
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 255
        frame[40:50, 40:50] = [0, 0, 255]  # 青い四角
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
        for num in [1, 3, 5, 10]:
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
    
    def test_linear_interpolation_midpoint(self, linear_interpolator, black_frame, white_frame):
        """線形補間の中間フレームが正しく計算されることを確認"""
        result = linear_interpolator.interpolate(black_frame, white_frame, num_frames=1)
        
        # 中間フレームは約 127-128 付近になるはず
        assert len(result) == 1
        expected_value = 127  # (0 + 255) / 2
        assert np.abs(result[0].mean() - expected_value) < 2
    
    def test_align_frames_same_size(self, interpolator):
        """同じサイズのフレームはそのまま返される"""
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)
        
        aligned1, aligned2 = interpolator._align_frames(frame1, frame2)
        
        assert aligned1.shape == (64, 64, 3)
        assert aligned2.shape == (64, 64, 3)
    
    def test_align_frames_different_size(self, interpolator):
        """異なるサイズのフレームは小さい方にリサイズされる"""
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((128, 128, 3), dtype=np.uint8)
        
        aligned1, aligned2 = interpolator._align_frames(frame1, frame2)
        
        assert aligned1.shape == (64, 64, 3)
        assert aligned2.shape == (64, 64, 3)
    
    def test_easing_linear(self, interpolator):
        """linear イージングが正しく動作することを確認"""
        result = interpolator._apply_easing(0.5, 'linear')
        assert result == 0.5
    
    def test_easing_ease_in(self, interpolator):
        """ease_in イージングが正しく動作することを確認"""
        result = interpolator._apply_easing(0.5, 'ease_in')
        assert result == 0.25  # 0.5 * 0.5
    
    def test_easing_ease_out(self, interpolator):
        """ease_out イージングが正しく動作することを確認"""
        result = interpolator._apply_easing(0.5, 'ease_out')
        assert result == 0.75  # 0.5 * (2 - 0.5)
    
    def test_easing_ease_in_out(self, interpolator):
        """ease_in_out イージングが正しく動作することを確認"""
        result_start = interpolator._apply_easing(0.25, 'ease_in_out')
        result_middle = interpolator._apply_easing(0.5, 'ease_in_out')
        result_end = interpolator._apply_easing(0.75, 'ease_in_out')
        
        assert result_start == 0.125  # 2 * 0.25 * 0.25
        assert result_middle == 0.5
        assert result_end == 0.875  # -1 + (4 - 2 * 0.75) * 0.75
    
    def test_interpolate_with_timing(self, interpolator, frame1, frame2):
        """interpolate_with_timing()が正しく動作することを確認"""
        result = interpolator.interpolate_with_timing(
            frame1, frame2, 
            num_frames=3, 
            easing='ease_in'
        )
        
        assert len(result) == 3
        for frame in result:
            assert frame.dtype == np.uint8
            assert frame.shape == frame1.shape
    
    def test_interpolate_with_morphing(self, morph_interpolator, frame1, frame2):
        """interpolate_with_morphing()が正しく動作することを確認"""
        result = morph_interpolator.interpolate_with_morphing(
            frame1, frame2,
            num_frames=3,
            use_feature_matching=False
        )
        
        assert len(result) == 3
        for frame in result:
            assert frame.dtype == np.uint8
    
    def test_rife_interpolation(self, interpolator, frame1, frame2):
        """RIFE補間が正しく動作することを確認"""
        result = interpolator._interpolate_rife(frame1, frame2, num_frames=3)
        
        assert len(result) == 3
        for frame in result:
            assert frame.dtype == np.uint8
            assert frame.shape == frame1.shape
    
    def test_tensor_conversion_roundtrip(self, interpolator, frame1):
        """NumPy -> Tensor -> NumPy の変換が正しく動作することを確認"""
        tensor = interpolator._numpy_to_tensor(frame1, 'cpu')
        result = interpolator._tensor_to_numpy(tensor)
        
        assert result.shape == frame1.shape
        assert result.dtype == np.uint8
        # 値が近いことを確認（浮動小数点誤差を許容）
        assert np.allclose(result, frame1, atol=2)

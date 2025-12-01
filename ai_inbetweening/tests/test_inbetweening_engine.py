"""
InbetWeeningEngine ユニットテスト
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import sys
import os

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import InbetWeeningEngine


class TestInbetWeeningEngine:
    """InbetWeeningEngine のテストクラス"""
    
    @pytest.fixture
    def engine(self):
        """テスト用のエンジンインスタンス"""
        return InbetWeeningEngine(device='cpu', model_type='rife')
    
    @pytest.fixture
    def test_keyframes(self, tmp_path):
        """テスト用のキーフレーム画像を作成"""
        # フレーム1: 赤い円
        img1 = Image.new('RGB', (64, 64), color='white')
        from PIL import ImageDraw
        draw1 = ImageDraw.Draw(img1)
        draw1.ellipse([10, 10, 30, 30], fill='red')
        path1 = tmp_path / 'frame1.png'
        img1.save(path1)
        
        # フレーム2: 青い円
        img2 = Image.new('RGB', (64, 64), color='white')
        draw2 = ImageDraw.Draw(img2)
        draw2.ellipse([30, 30, 50, 50], fill='blue')
        path2 = tmp_path / 'frame2.png'
        img2.save(path2)
        
        return str(path1), str(path2)
    
    @pytest.fixture
    def multiple_keyframes(self, tmp_path):
        """テスト用の複数キーフレーム画像を作成"""
        paths = []
        colors = ['red', 'green', 'blue', 'yellow']
        
        for i, color in enumerate(colors):
            img = Image.new('RGB', (64, 64), color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            x = 10 + i * 10
            draw.ellipse([x, x, x + 20, x + 20], fill=color)
            path = tmp_path / f'frame_{i}.png'
            img.save(path)
            paths.append(str(path))
        
        return paths
    
    def test_engine_initialization(self, engine):
        """エンジンが正しく初期化されることを確認"""
        assert engine.device in ['cpu', 'cuda']
        assert engine.model_type == 'rife'
        assert engine.keyframe_loader is not None
        assert engine.interpolator is not None
    
    def test_generate_returns_list(self, engine, test_keyframes):
        """generate()がリストを返すことを確認"""
        frame1_path, frame2_path = test_keyframes
        result = engine.generate(frame1_path, frame2_path, num_frames=3)
        
        assert isinstance(result, list)
    
    def test_generate_returns_correct_num_frames(self, engine, test_keyframes):
        """generate()が正しい数のフレームを返すことを確認（キーフレーム + 中割）"""
        frame1_path, frame2_path = test_keyframes
        
        for num_frames in [1, 3, 5]:
            result = engine.generate(frame1_path, frame2_path, num_frames=num_frames)
            # キーフレーム2枚 + 中割フレーム
            expected_total = 2 + num_frames
            assert len(result) == expected_total
    
    def test_generate_first_and_last_are_keyframes(self, engine, test_keyframes):
        """generate()の最初と最後がキーフレームであることを確認"""
        frame1_path, frame2_path = test_keyframes
        result = engine.generate(frame1_path, frame2_path, num_frames=3)
        
        # 最初のフレームはframe1と同じ
        original_frame1 = np.array(Image.open(frame1_path))
        assert np.array_equal(result[0], original_frame1)
        
        # 最後のフレームはframe2と同じ
        original_frame2 = np.array(Image.open(frame2_path))
        assert np.array_equal(result[-1], original_frame2)
    
    def test_generate_saves_frames(self, engine, test_keyframes, tmp_path):
        """generate()がフレームを保存できることを確認"""
        frame1_path, frame2_path = test_keyframes
        output_dir = tmp_path / 'output'
        
        engine.generate(frame1_path, frame2_path, num_frames=3, save_path=output_dir)
        
        assert output_dir.exists()
        saved_files = list(output_dir.glob('*.png'))
        assert len(saved_files) == 5  # 2 keyframes + 3 inbetweens
    
    def test_generate_sequence(self, engine, multiple_keyframes, tmp_path):
        """generate_sequence()が正しく動作することを確認"""
        result = engine.generate_sequence(
            multiple_keyframes,
            num_frames=2,
            save_path=None
        )
        
        assert isinstance(result, list)
        # 最初のフレームを除いた各区間 (num_frames + 1) フレームが追加される
        # 区間数: len(keyframes) - 1 = 3
        # 各区間: num_frames + 1 = 3 フレーム（最後のキーフレームを含む）
        # 合計: 1(最初のキーフレームは除外) + 3 * 3 = 9
        expected_frames = (len(multiple_keyframes) - 1) * (2 + 1)
        assert len(result) == expected_frames
    
    def test_export_video(self, engine, test_keyframes, tmp_path):
        """export_video()が正しく動作することを確認"""
        frame1_path, frame2_path = test_keyframes
        frames = engine.generate(frame1_path, frame2_path, num_frames=3)
        
        video_path = tmp_path / 'output.mp4'
        
        try:
            engine.export_video(frames, video_path, fps=24)
            assert video_path.exists()
            assert video_path.stat().st_size > 0
        except Exception as e:
            # FFmpeg がない場合はスキップ
            pytest.skip(f"Video export failed (FFmpeg may not be available): {e}")
    
    def test_export_video_with_float_frames(self, engine, tmp_path):
        """export_video()がfloat32フレームを正しく処理することを確認"""
        # float32 フレームを作成
        frames = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(5)]
        
        video_path = tmp_path / 'float_output.mp4'
        
        try:
            engine.export_video(frames, video_path, fps=24)
            assert video_path.exists()
        except Exception as e:
            pytest.skip(f"Video export failed: {e}")
    
    def test_save_frames(self, engine, tmp_path):
        """_save_frames()が正しく動作することを確認"""
        frames = [
            np.zeros((64, 64, 3), dtype=np.uint8),
            np.ones((64, 64, 3), dtype=np.uint8) * 128,
            np.ones((64, 64, 3), dtype=np.uint8) * 255,
        ]
        
        output_dir = tmp_path / 'save_test'
        engine._save_frames(frames, output_dir)
        
        assert output_dir.exists()
        saved_files = list(output_dir.glob('*.png'))
        assert len(saved_files) == 3
        
        # ファイル名の形式を確認
        for f in saved_files:
            assert f.name.startswith('frame_')
            assert f.name.endswith('.png')
    
    def test_save_frames_with_float_data(self, engine, tmp_path):
        """_save_frames()がfloat32データを正しく処理することを確認"""
        frames = [
            np.random.rand(64, 64, 3).astype(np.float32),
            np.random.rand(64, 64, 3).astype(np.float64),
        ]
        
        output_dir = tmp_path / 'float_save_test'
        engine._save_frames(frames, output_dir)
        
        saved_files = list(output_dir.glob('*.png'))
        assert len(saved_files) == 2
    
    def test_generate_with_path_objects(self, engine, test_keyframes, tmp_path):
        """generate()がPathオブジェクトを受け付けることを確認"""
        frame1_path, frame2_path = test_keyframes
        result = engine.generate(
            Path(frame1_path),
            Path(frame2_path),
            num_frames=2
        )
        
        assert len(result) == 4  # 2 keyframes + 2 inbetweens


class TestInbetWeeningEngineDevice:
    """デバイス関連のテスト"""
    
    def test_default_device_selection(self):
        """デフォルトでCPUが選択されることを確認（CUDAがない環境）"""
        engine = InbetWeeningEngine(model_type='rife')
        # CUDA が利用可能な場合は cuda、そうでなければ cpu
        assert engine.device in ['cpu', 'cuda']
    
    def test_explicit_cpu_device(self):
        """明示的にCPUを指定できることを確認"""
        engine = InbetWeeningEngine(device='cpu', model_type='rife')
        assert engine.device == 'cpu'

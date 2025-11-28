#!/usr/bin/env python3
"""
テスト用ダミーキーフレーム画像生成スクリプト
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import InbetWeeningEngine


def create_test_frames():
    """テスト用のダミーキーフレーム画像を生成"""
    
    width, height = 256, 256
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # フレーム1: 左上の円
    img1 = Image.new('RGB', (width, height), color='white')
    draw1 = ImageDraw.Draw(img1)
    draw1.ellipse([50, 50, 100, 100], fill='red')
    img1.save(data_dir / 'keyframe1.png')
    print(f"✓ Created: {data_dir / 'keyframe1.png'}")
    
    # フレーム2: 右下の円
    img2 = Image.new('RGB', (width, height), color='white')
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([150, 150, 200, 200], fill='blue')
    img2.save(data_dir / 'keyframe2.png')
    print(f"✓ Created: {data_dir / 'keyframe2.png'}")
    
    return str(data_dir / 'keyframe1.png'), str(data_dir / 'keyframe2.png')


def main():
    """テスト実行"""
    
    print("=" * 60)
    print("AI Inbetweening System - Test")
    print("=" * 60)
    
    # テスト用画像を生成
    print("\nGenerating test frames...")
    frame1_path, frame2_path = create_test_frames()
    
    # エンジンの初期化と実行
    print("\nInitializing engine...")
    engine = InbetWeeningEngine(device='cpu', model_type='rife')
    
    print("\nGenerating inbetween frames...")
    output_dir = Path(__file__).parent / 'output'
    
    try:
        frames = engine.generate(
            frame1_path,
            frame2_path,
            num_frames=5,
            save_path=output_dir
        )
        
        print(f"\n✓ Successfully generated {len(frames)} frames!")
        print(f"  Output directory: {output_dir}")
        
        # 動画にもエクスポート
        video_path = output_dir / 'inbetweening_demo.mp4'
        engine.export_video(frames, video_path, fps=12)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

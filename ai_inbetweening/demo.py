"""
AI Inbetweening System - デモスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import InbetWeeningEngine


def main():
    """メインデモ関数"""
    
    print("=" * 60)
    print("AI Inbetweening System - Demo")
    print("=" * 60)
    
    # エンジンの初期化
    engine = InbetWeeningEngine(device='cpu', model_type='rife')
    
    print("\n✓ Engine initialized successfully!")
    print(f"  Device: {engine.device}")
    print(f"  Model: {engine.model_type}")
    
    print("\n" + "=" * 60)
    print("使用方法:")
    print("=" * 60)
    
    print("""
1. キーフレーム2枚から中割を生成:
   
   frames = engine.generate(
       'keyframe1.png',
       'keyframe2.png',
       num_frames=5,
       save_path='output/'
   )

2. 複数のキーフレームからシーケンスを生成:
   
   all_frames = engine.generate_sequence(
       ['frame1.png', 'frame2.png', 'frame3.png'],
       num_frames=5,
       save_path='output/'
   )

3. 動画ファイルにエクスポート:
   
   engine.export_video(frames, 'output.mp4', fps=24)
    """)
    
    print("\n" + "=" * 60)
    print("次のステップ:")
    print("=" * 60)
    print("""
1. テスト用のキーフレーム画像を data/ に配置
2. generate() メソッドでフレームを生成
3. 出力を確認
    """)


if __name__ == '__main__':
    main()

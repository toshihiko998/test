# AI Inbetweening System

キーフレーム画像からアニメの中割（インビトウィーン）をAIで自動生成するシステムです。

## 機能

- キーフレーム画像の入力
- AI による中割フレーム生成
- 複数フレーム補間
- 出力画像のエクスポート

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```python
from src.inbetweening_engine import InbetWeeningEngine

engine = InbetWeeningEngine()
output_frames = engine.generate(keyframe1_path, keyframe2_path, num_frames=5)
```

## プロジェクト構成

```
ai_inbetweening/
├── src/
│   ├── __init__.py
│   ├── inbetweening_engine.py      # メインエンジン
│   ├── keyframe_loader.py          # キーフレーム入力
│   └── frame_interpolation.py      # フレーム補間
├── models/                          # 学習済みモデル保存先
├── data/                            # テスト用データ
├── output/                          # 出力先
├── requirements.txt
└── setup.py
```

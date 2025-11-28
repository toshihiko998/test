# AI Inbetweening System - セットアップ＆実行ガイド

## セットアップ

### 1. 依存パッケージのインストール

```bash
cd /workspaces/test/ai_inbetweening
pip install -r requirements.txt
```

### 2. サーバーの起動

```bash
python app.py
```

出力:
```
============================================================
AI Inbetweening System - Server Starting
============================================================
Open http://localhost:5000 in your browser
============================================================
```

### 3. ブラウザでアクセス

- **ローカル環境**: `http://localhost:5000`
- **リモート環境**: `http://<リモートホスト>:5000`

## 使用方法

### Web UI での使用

1. **キーフレーム1（開始画像）** をアップロード
2. **キーフレーム2（終了画像）** をアップロード
3. **中割フレーム数**を指定（2～30）
4. **出力FPS**を指定（15～60）
5. **「🚀 中割を生成」** ボタンをクリック
6. 動画ファイルが自動的にダウンロードされます

### Python スクリプトでの使用

```python
from src import InbetWeeningEngine

# エンジンを初期化
engine = InbetWeeningEngine(device='cpu', model_type='rife')

# フレームを生成
frames = engine.generate(
    'keyframe1.png',
    'keyframe2.png',
    num_frames=5,
    save_path='output/'
)

# 動画をエクスポート
engine.export_video(frames, 'output.mp4', fps=24)
```

## トラブルシューティング

### 「Cannot GET /」エラーが出る

サーバーが起動していないか、ポート5000が埋まっている可能性があります。

**解決方法:**

```bash
# プロセスの確認
lsof -i :5000

# プロセスを終了（PIDは上記コマンドで確認）
kill -9 <PID>

# サーバーを再起動
python app.py
```

### モデルダウンロードエラー

初回実行時、RIFEモデルが自動ダウンロードされます。インターネット接続を確認してください。

### メモリ不足エラー

CUDA（GPU）を使用している場合はGPUメモリが不足している可能性があります。

**解決方法:**

```python
# CPUを使用するよう変更
engine = InbetWeeningEngine(device='cpu', model_type='rife')
```

## ディレクトリ構成

```
ai_inbetweening/
├── app.py                 # Flask API サーバー
├── demo.py               # デモスクリプト
├── requirements.txt      # 依存パッケージ
├── setup.py             # セットアップスクリプト
├── test_demo.py         # テストコード
├── data/                # サンプル画像フォルダ
├── models/              # モデルの保存フォルダ
├── output/              # 出力フォルダ
├── uploads/             # アップロード一時ファイル
└── src/
    ├── __init__.py
    ├── frame_interpolation.py
    ├── inbetweening_engine.py
    └── keyframe_loader.py
```

## API エンドポイント

### GET `/`
ホームページ（Web UI）を返す

**レスポンス**: HTML

### POST `/generate`
キーフレーム画像から中割を生成し、動画をダウンロード

**リクエスト:**
```
Content-Type: multipart/form-data
- frame1: Image file (required)
- frame2: Image file (required)
- num_frames: Integer (2-30, default: 4)
- fps: Integer (15-60, default: 24)
```

**レスポンス**: MP4 動画ファイル

### GET `/health`
サーバーのヘルスチェック

**レスポンス:**
```json
{"status": "ok"}
```

## 注意事項

- 大きな画像は自動的にリサイズされます
- 処理時間はマシンスペックに依存します（通常数秒～数分）
- 出力動画の品質は入力画像の品質に依存します

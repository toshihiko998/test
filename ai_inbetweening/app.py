"""
AI Inbetweening System - Flask API Server
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import InbetWeeningEngine

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# アップロードフォルダを作成
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# グローバルエンジン
engine = None

def init_engine():
    """エンジンを初期化"""
    global engine
    if engine is None:
        device = 'cuda' if os.environ.get('CUDA_AVAILABLE') == '1' else 'cpu'
        engine = InbetWeeningEngine(device=device, model_type='rife')
        print(f"✓ Engine initialized on {device}")


@app.route('/')
def index():
    """ホームページ"""
    return '''
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Inbetweening System</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 10px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 28px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .form-group {
                margin-bottom: 25px;
            }
            label {
                display: block;
                color: #333;
                font-weight: 600;
                margin-bottom: 8px;
                font-size: 14px;
            }
            input[type="file"],
            select,
            input[type="number"] {
                width: 100%;
                padding: 10px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            input[type="file"]:focus,
            select:focus,
            input[type="number"]:focus {
                outline: none;
                border-color: #667eea;
            }
            .file-input-wrapper {
                position: relative;
                overflow: hidden;
                display: inline-block;
                width: 100%;
            }
            input[type="file"] {
                position: absolute;
                left: -9999px;
            }
            .file-input-label {
                display: block;
                padding: 30px;
                background: #f5f5f5;
                border: 2px dashed #667eea;
                border-radius: 5px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                color: #667eea;
                font-weight: 600;
            }
            .file-input-label:hover {
                background: #f0f0ff;
                border-color: #764ba2;
            }
            .file-inputs {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }
            .file-input-group {
                flex: 1;
            }
            .file-input-group label {
                font-size: 13px;
                margin-bottom: 6px;
            }
            .params {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }
            button {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                margin-top: 20px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .message {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .message.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
                display: block;
            }
            .message.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                display: block;
            }
            .message.loading {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
                display: block;
            }
            .progress-bar {
                width: 100%;
                height: 5px;
                background: #e0e0e0;
                border-radius: 5px;
                overflow: hidden;
                margin-top: 10px;
                display: none;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                width: 0%;
                animation: progress 2s infinite;
            }
            @keyframes progress {
                0% { width: 0%; }
                50% { width: 100%; }
                100% { width: 100%; }
            }
            .file-name {
                margin-top: 5px;
                font-size: 12px;
                color: #666;
                word-break: break-all;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 AI Inbetweening System</h1>
            <p class="subtitle">キーフレーム画像から自動で中割を生成します</p>
            
            <form id="uploadForm">
                <div class="file-inputs">
                    <div class="file-input-group">
                        <label for="frame1">キーフレーム1 (開始画像)</label>
                        <div class="file-input-wrapper">
                            <input type="file" id="frame1" name="frame1" accept="image/*" required>
                            <label for="frame1" class="file-input-label">
                                📁 ファイルを選択
                            </label>
                        </div>
                        <div class="file-name" id="frame1-name"></div>
                    </div>
                    <div class="file-input-group">
                        <label for="frame2">キーフレーム2 (終了画像)</label>
                        <div class="file-input-wrapper">
                            <input type="file" id="frame2" name="frame2" accept="image/*" required>
                            <label for="frame2" class="file-input-label">
                                📁 ファイルを選択
                            </label>
                        </div>
                        <div class="file-name" id="frame2-name"></div>
                    </div>
                </div>

                <div class="form-group">
                    <label for="numFrames">中割フレーム数 (2-30)</label>
                    <input type="number" id="numFrames" name="numFrames" value="4" min="2" max="30">
                </div>

                <div class="form-group">
                    <label for="fps">出力FPS (15-60)</label>
                    <input type="number" id="fps" name="fps" value="24" min="15" max="60">
                </div>

                <button type="submit" id="submitBtn">🚀 中割を生成</button>
            </form>

            <div class="progress-bar" id="progressBar">
                <div class="progress-fill"></div>
            </div>

            <div class="message" id="message"></div>
        </div>

        <script>
            // ファイル入力の表示名を更新
            document.getElementById('frame1').addEventListener('change', function(e) {
                document.getElementById('frame1-name').textContent = e.target.files[0]?.name || '';
            });
            document.getElementById('frame2').addEventListener('change', function(e) {
                document.getElementById('frame2-name').textContent = e.target.files[0]?.name || '';
            });

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const frame1 = document.getElementById('frame1').files[0];
                const frame2 = document.getElementById('frame2').files[0];
                const numFrames = document.getElementById('numFrames').value;
                const fps = document.getElementById('fps').value;

                if (!frame1 || !frame2) {
                    showMessage('両方のキーフレーム画像を選択してください', 'error');
                    return;
                }

                const formData = new FormData();
                formData.append('frame1', frame1);
                formData.append('frame2', frame2);
                formData.append('num_frames', numFrames);
                formData.append('fps', fps);

                showMessage('処理中... 少々お待ちください...', 'loading');
                document.getElementById('progressBar').style.display = 'block';
                document.getElementById('submitBtn').disabled = true;

                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'inbetweening.mp4';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                        showMessage('✅ 生成完了！動画をダウンロードしました', 'success');
                    } else {
                        const error = await response.text();
                        showMessage('❌ エラー: ' + error, 'error');
                    }
                } catch (error) {
                    showMessage('❌ 通信エラー: ' + error.message, 'error');
                } finally {
                    document.getElementById('progressBar').style.display = 'none';
                    document.getElementById('submitBtn').disabled = false;
                }
            });

            function showMessage(text, type) {
                const msgEl = document.getElementById('message');
                msgEl.textContent = text;
                msgEl.className = 'message ' + type;
                setTimeout(() => {
                    if (type !== 'loading') msgEl.style.display = 'none';
                }, type === 'error' ? 5000 : 3000);
            }
        </script>
    </body>
    </html>
    '''


@app.route('/generate', methods=['POST'])
def generate():
    """フレームを生成"""
    global engine
    
    try:
        # ファイルの取得
        if 'frame1' not in request.files or 'frame2' not in request.files:
            return 'フレーム画像が見つかりません', 400
        
        frame1_file = request.files['frame1']
        frame2_file = request.files['frame2']
        
        if frame1_file.filename == '' or frame2_file.filename == '':
            return 'ファイルが選択されていません', 400
        
        num_frames = int(request.form.get('num_frames', 4))
        fps = int(request.form.get('fps', 24))
        
        # 入力範囲チェック
        if num_frames < 2 or num_frames > 30:
            return 'フレーム数は2～30の範囲で指定してください', 400
        
        if fps < 15 or fps > 60:
            return 'FPSは15～60の範囲で指定してください', 400
        
        # ファイルを一時保存
        frame1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('frame1_' + frame1_file.filename))
        frame2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('frame2_' + frame2_file.filename))
        
        frame1_file.save(frame1_path)
        frame2_file.save(frame2_path)
        
        # フレームを生成
        init_engine()
        frames = engine.generate(
            frame1_path,
            frame2_path,
            num_frames=num_frames,
            save_path=app.config['OUTPUT_FOLDER']
        )
        
        # 動画をエクスポート
        video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
        engine.export_video(frames, video_path, fps=fps)
        
        # ファイルを送信
        return send_file(video_path, mimetype='video/mp4', as_attachment=True, download_name='inbetweening.mp4')
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return f'エラーが発生しました: {str(e)}', 500
    finally:
        # 一時ファイルを削除
        try:
            if os.path.exists(frame1_path):
                os.remove(frame1_path)
            if os.path.exists(frame2_path):
                os.remove(frame2_path)
        except:
            pass


@app.route('/health')
def health():
    """ヘルスチェック"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AI Inbetweening System - Server Starting")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

"""
AI Inbetweening System - Flask API Server
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
from datetime import datetime
from urllib.parse import quote
import json
import atexit
from threading import Timer

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# グローバル削除スケジューラー
_delete_timers = []

def _schedule_deletion(path_str: str, delay: int = 2):
    """ファイル/ディレクトリの削除をスケジュール"""
    def delete_path():
        try:
            path = Path(path_str)
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                print(f"✓ Auto-deleted directory: {path}")
            elif path.is_file():
                path.unlink()
                print(f"✓ Auto-deleted file: {path}")
        except Exception as e:
            print(f"⚠ Failed to auto-delete {path_str}: {e}")
    
    timer = Timer(delay, delete_path)
    timer.daemon = True
    timer.start()
    _delete_timers.append(timer)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

project_root = Path(__file__).parent
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')

# 環境変数で出力フォルダを指定可能
output_base = os.environ.get('OUTPUT_BASE_PATH')
if output_base and output_base.strip():
    app.config['OUTPUT_FOLDER'] = output_base.strip()
else:
    app.config['OUTPUT_FOLDER'] = os.path.join(project_root, 'output')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

print(f"✓ OUTPUT_FOLDER: {app.config['OUTPUT_FOLDER']}")


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
            * { margin: 0; padding: 0; box-sizing: border-box; }
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
            h1 { color: #333; margin-bottom: 10px; font-size: 28px; }
            .subtitle { color: #666; margin-bottom: 30px; font-size: 14px; }
            .form-group { margin-bottom: 25px; }
            label {
                display: block;
                color: #333;
                font-weight: 600;
                margin-bottom: 8px;
                font-size: 14px;
            }
            input[type="file"], select, input[type="number"], input[type="text"] {
                width: 100%;
                padding: 10px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            input[type="checkbox"] {
                width: auto;
                margin-right: 8px;
            }
            label input[type="checkbox"] {
                cursor: pointer;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            .file-inputs {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }
            .file-input-group { flex: 1; }
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
            button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4); }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            .message { margin-top: 20px; padding: 15px; border-radius: 5px; display: none; }
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
            .message.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
                display: block;
            }
            .file-name { margin-top: 5px; font-size: 12px; color: #666; }
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
                        <input type="file" id="frame1" name="frame1" accept="image/*" required>
                        <div class="file-name" id="frame1-name"></div>
                    </div>
                    <div class="file-input-group">
                        <label for="frame2">キーフレーム2 (終了画像)</label>
                        <input type="file" id="frame2" name="frame2" accept="image/*" required>
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

                <div class="form-group">
                    <label for="outputSubdir">出力保存先サブディレクトリ（任意）</label>
                    <input type="text" id="outputSubdir" name="output_subdir" placeholder="例: my_run_001">
                </div>

                <div class="form-group">
                    <label>
                        <input type="checkbox" id="useMorphing" name="use_morphing" checked>
                        ✨ 高度なポーズ・スケール補間を使用（時間がかかります）
                    </label>
                </div>

                <button type="submit" id="submitBtn">🚀 中割を生成</button>
            </form>

            <div class="message" id="message"></div>
        </div>

        <script>
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
                const outputSubdir = document.getElementById('outputSubdir')?.value || '';
                const useMorphing = document.getElementById('useMorphing')?.checked ? 'true' : 'false';

                if (!frame1 || !frame2) {
                    showMessage('両方のキーフレーム画像を選択してください', 'error');
                    return;
                }

                const formData = new FormData();
                formData.append('frame1', frame1);
                formData.append('frame2', frame2);
                formData.append('num_frames', numFrames);
                formData.append('fps', fps);
                formData.append('output_subdir', outputSubdir);
                formData.append('use_morphing', useMorphing);

                showMessage('処理中... 少々お待ちください...', 'info');
                document.getElementById('submitBtn').disabled = true;

                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        if (data && data.list_url) {
                            window.location.href = data.list_url;
                        } else {
                            showMessage('✅ 生成完了しました', 'success');
                        }
                    } else {
                        showMessage('❌ エラー: ' + (data?.error || 'Unknown error'), 'error');
                    }
                } catch (error) {
                    showMessage('❌ 通信エラー: ' + error.message, 'error');
                } finally {
                    document.getElementById('submitBtn').disabled = false;
                }
            });

            function showMessage(text, type) {
                const msgEl = document.getElementById('message');
                msgEl.textContent = text;
                msgEl.className = 'message ' + type;
            }
        </script>
    </body>
    </html>
    '''


@app.route('/generate', methods=['POST'])
def generate():
    """フレームを生成"""
    
    try:
        if 'frame1' not in request.files or 'frame2' not in request.files:
            return jsonify({'error': 'フレーム画像が見つかりません'}), 400
        
        frame1_file = request.files['frame1']
        frame2_file = request.files['frame2']
        
        if frame1_file.filename == '' or frame2_file.filename == '':
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        num_frames = int(request.form.get('num_frames', 4))
        fps = int(request.form.get('fps', 24))
        output_subdir = (request.form.get('output_subdir') or '').strip()
        use_morphing = request.form.get('use_morphing', 'true').lower() == 'true'
        
        if num_frames < 2 or num_frames > 30:
            return jsonify({'error': 'フレーム数は2～30の範囲で指定してください'}), 400
        
        if fps < 15 or fps > 60:
            return jsonify({'error': 'FPSは15～60の範囲で指定してください'}), 400
        
        # 保存先を決定
        if output_subdir:
            safe_subdir = secure_filename(output_subdir)
            save_dir = os.path.join(app.config['OUTPUT_FOLDER'], safe_subdir)
        else:
            save_dir = app.config['OUTPUT_FOLDER']
        
        os.makedirs(save_dir, exist_ok=True)
        
        # ファイルを一時保存
        frame1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('frame1_' + frame1_file.filename))
        frame2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('frame2_' + frame2_file.filename))
        
        frame1_file.save(frame1_path)
        frame2_file.save(frame2_path)
        
        # ここから実際のエンジン処理
        from src import InbetWeeningEngine
        engine = InbetWeeningEngine(device='cpu', model_type='rife')
        
        # 画像読み込み
        from PIL import Image
        import numpy as np
        
        frame1 = np.array(Image.open(frame1_path).convert('RGB'))
        frame2 = np.array(Image.open(frame2_path).convert('RGB'))
        
        # モーフィングまたは通常の補間を選択
        interpolator = engine.interpolator
        if use_morphing:
            print("✓ Using morphing interpolation with pose/scale aware features")
            frames = interpolator.interpolate_with_morphing(frame1, frame2, num_frames, use_feature_matching=True)
        else:
            print("✓ Using standard optical flow interpolation")
            frames = engine.generate(
                frame1_path,
                frame2_path,
                num_frames=num_frames,
                save_path=None
            )
        
        # タイムスタンプ付きで保存
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for idx, frame in enumerate(frames):
            if hasattr(frame, 'dtype') and (frame.dtype == 'float32' or frame.dtype == 'float64'):
                frame_to_save = (frame * 255).astype('uint8')
            else:
                frame_to_save = frame
            img = Image.fromarray(frame_to_save)
            out_name = os.path.join(save_dir, f"{ts}_frame_{idx:04d}.png")
            img.save(out_name)
        
        # 動画をエクスポート
        video_name = f"{ts}_output.mp4"
        video_path = os.path.join(save_dir, video_name)
        engine.export_video(frames, video_path, fps=fps)
        
        # 一覧ページのURLを返す
        list_url = f"/files?dir={quote(save_dir)}"
        return jsonify({'status': 'ok', 'list_url': list_url})
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'エラーが発生しました: {str(e)}'}), 500
    
    finally:
        try:
            if os.path.exists(frame1_path):
                os.remove(frame1_path)
            if os.path.exists(frame2_path):
                os.remove(frame2_path)
        except:
            pass


@app.route('/files')
def files():
    """ファイル一覧表示"""
    dir_param = request.args.get('dir')
    if not dir_param:
        return "dir パラメータを指定してください", 400
    
    target_dir = Path(dir_param)
    
    if not target_dir.exists() or not target_dir.is_dir():
        return f"ディレクトリが存在しません: {target_dir}", 404
    
    files_list = sorted(target_dir.iterdir(), key=lambda p: p.name)
    html = ["<html><head><meta charset=\"utf-8\"><title>ファイル一覧</title>"]
    html.append("<style>")
    html.append("body { font-family: Arial; margin: 20px; }")
    html.append("table { border-collapse: collapse; width: 100%; margin-top: 20px; }")
    html.append("th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }")
    html.append("th { background-color: #667eea; color: white; }")
    html.append("a { color: #667eea; text-decoration: none; }")
    html.append("a:hover { text-decoration: underline; }")
    html.append(".btn-group { margin: 20px 0; }")
    html.append(".btn { display: inline-block; padding: 10px 20px; margin-right: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 5px; }")
    html.append(".btn:hover { opacity: 0.9; }")
    html.append("</style></head><body>")
    html.append(f"<h2>保存先: {target_dir}</h2>")
    
    # 一括ダウンロードボタン
    download_url = f"/download?path={quote(str(target_dir))}"
    html.append('<div class="btn-group">')
    html.append(f'<a href="{download_url}" class="btn">📦 全ファイルをZIPでダウンロード</a>')
    html.append('</div>')
    
    html.append("<table><tr><th>ファイル名</th><th>サイズ</th><th>操作</th></tr>")
    
    for f in files_list:
        name = f.name
        size = f.stat().st_size if f.is_file() else '-'
        size_str = f"{size / (1024*1024):.2f} MB" if isinstance(size, int) and size > 0 else str(size)
        href = f"/download?path={quote(str(f))}"
        html.append(f"<tr><td>{name}</td><td>{size_str}</td><td><a href=\"{href}\">ダウンロード</a></td></tr>")
    
    html.append("</table></body></html>")
    return '\n'.join(html)


@app.route('/download')
def download():
    """ファイルダウンロード or ZIP 一括ダウンロード（ダウンロード後に自動削除）"""
    path_param = request.args.get('path')
    if not path_param:
        return "path パラメータを指定してください", 400
    
    target = Path(path_param)
    
    # ディレクトリの場合は ZIP を生成してからディレクトリを削除
    if target.is_dir():
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in sorted(target.iterdir()):
                if file.is_file():
                    arcname = f"{target.name}/{file.name}"
                    zip_file.write(file, arcname=arcname)
        
        zip_buffer.seek(0)
        
        # ダウンロード完了後にディレクトリを削除するようスケジュール
        _schedule_deletion(str(target), delay=2)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{target.name}.zip"
        )
    
    # ファイルの場合は直接ダウンロード
    if not target.exists() or not target.is_file():
        return f"ファイルが存在しません: {target}", 404
    
    # ダウンロード完了後にファイルを削除するようスケジュール
    _schedule_deletion(str(target), delay=2)
    
    return send_file(str(target), as_attachment=True)


@app.route('/health')
def health():
    """ヘルスチェック"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    
    print("\n" + "=" * 60)
    print("AI Inbetweening System - Lightweight Server")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"URL: http://localhost:{port}")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

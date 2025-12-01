"""
Flask API ユニットテスト
"""

import pytest
from pathlib import Path
from io import BytesIO
from PIL import Image

from app import app as flask_app


@pytest.fixture
def app():
    """Flaskアプリケーションのテスト用フィクスチャ"""
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    return flask_app


@pytest.fixture
def client(app):
    """テストクライアント"""
    return app.test_client()


@pytest.fixture
def test_image():
    """テスト用の画像データを生成"""
    img = Image.new('RGB', (64, 64), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def test_image2():
    """テスト用の画像データ2を生成"""
    img = Image.new('RGB', (64, 64), color='blue')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


class TestFlaskApp:
    """Flask アプリのテストクラス"""
    
    def test_index_route(self, client):
        """ホームページが正しく表示されることを確認"""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'AI Inbetweening System' in response.data
    
    def test_index_contains_form(self, client):
        """ホームページにフォームが含まれることを確認"""
        response = client.get('/')
        
        assert b'<form' in response.data
        assert b'frame1' in response.data
        assert b'frame2' in response.data
        assert b'numFrames' in response.data
        assert b'fps' in response.data
    
    def test_health_route(self, client):
        """ヘルスチェックエンドポイントが正しく動作することを確認"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
    
    def test_generate_missing_files(self, client):
        """ファイルなしでgenerateを呼び出すとエラーになることを確認"""
        response = client.post('/generate')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_generate_missing_frame2(self, client, test_image):
        """frame2なしでgenerateを呼び出すとエラーになることを確認"""
        response = client.post('/generate', data={
            'frame1': (test_image, 'frame1.png'),
        })
        
        assert response.status_code == 400
    
    def test_generate_empty_filenames(self, client):
        """空のファイル名でgenerateを呼び出すとエラーになることを確認"""
        response = client.post('/generate', data={
            'frame1': (BytesIO(), ''),
            'frame2': (BytesIO(), ''),
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_generate_invalid_num_frames_low(self, client, test_image, test_image2):
        """num_framesが小さすぎるとエラーになることを確認"""
        response = client.post('/generate', data={
            'frame1': (test_image, 'frame1.png'),
            'frame2': (test_image2, 'frame2.png'),
            'num_frames': '1',  # 2未満はエラー
            'fps': '24',
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_generate_invalid_num_frames_high(self, client, test_image, test_image2):
        """num_framesが大きすぎるとエラーになることを確認"""
        # 画像データを再作成（BytesIOは1回しか読めないため）
        img1 = Image.new('RGB', (64, 64), color='red')
        buffer1 = BytesIO()
        img1.save(buffer1, format='PNG')
        buffer1.seek(0)
        
        img2 = Image.new('RGB', (64, 64), color='blue')
        buffer2 = BytesIO()
        img2.save(buffer2, format='PNG')
        buffer2.seek(0)
        
        response = client.post('/generate', data={
            'frame1': (buffer1, 'frame1.png'),
            'frame2': (buffer2, 'frame2.png'),
            'num_frames': '31',  # 30を超えるとエラー
            'fps': '24',
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_generate_invalid_fps_low(self, client, test_image, test_image2):
        """fpsが小さすぎるとエラーになることを確認"""
        img1 = Image.new('RGB', (64, 64), color='red')
        buffer1 = BytesIO()
        img1.save(buffer1, format='PNG')
        buffer1.seek(0)
        
        img2 = Image.new('RGB', (64, 64), color='blue')
        buffer2 = BytesIO()
        img2.save(buffer2, format='PNG')
        buffer2.seek(0)
        
        response = client.post('/generate', data={
            'frame1': (buffer1, 'frame1.png'),
            'frame2': (buffer2, 'frame2.png'),
            'num_frames': '4',
            'fps': '10',  # 15未満はエラー
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_generate_invalid_fps_high(self, client, test_image, test_image2):
        """fpsが大きすぎるとエラーになることを確認"""
        img1 = Image.new('RGB', (64, 64), color='red')
        buffer1 = BytesIO()
        img1.save(buffer1, format='PNG')
        buffer1.seek(0)
        
        img2 = Image.new('RGB', (64, 64), color='blue')
        buffer2 = BytesIO()
        img2.save(buffer2, format='PNG')
        buffer2.seek(0)
        
        response = client.post('/generate', data={
            'frame1': (buffer1, 'frame1.png'),
            'frame2': (buffer2, 'frame2.png'),
            'num_frames': '4',
            'fps': '65',  # 60を超えるとエラー
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data


class TestFilesEndpoint:
    """/files エンドポイントのテスト"""
    
    def test_files_missing_dir_param(self, client):
        """dirパラメータなしでfilesを呼び出すとエラーになることを確認"""
        response = client.get('/files')
        
        assert response.status_code == 400
    
    def test_files_nonexistent_dir(self, client):
        """存在しないディレクトリを指定するとエラーになることを確認"""
        response = client.get('/files?dir=/nonexistent/path')
        
        # 絶対パス制限により403または404
        assert response.status_code in [403, 404]


class TestDownloadEndpoint:
    """/download エンドポイントのテスト"""
    
    def test_download_missing_path_param(self, client):
        """pathパラメータなしでdownloadを呼び出すとエラーになることを確認"""
        response = client.get('/download')
        
        assert response.status_code == 400
    
    def test_download_nonexistent_file(self, client):
        """存在しないファイルを指定するとエラーになることを確認"""
        response = client.get('/download?path=/nonexistent/file.png')
        
        # 絶対パス制限により403または404
        assert response.status_code in [403, 404]

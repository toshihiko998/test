"""
シンプルなテスト用 Flask サーバー
"""
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Inbetweening Test</title>
    </head>
    <body>
        <h1>✅ Server is running!</h1>
        <p>Flask サーバーが正常に動作しています。</p>
    </body>
    </html>
    '''

@app.route('/health')
def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    print("Starting test server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

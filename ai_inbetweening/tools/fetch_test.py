"""
Simple fetch-like tester for /generate endpoint.
Tries to POST two test images and then attempts to parse JSON like browser's response.json().
Prints raw response and exception to help reproduce "Unexpected end of JSON input".
"""
import requests
from pathlib import Path

URL = "http://127.0.0.1:8000/generate"
HERE = Path(__file__).parent.parent
FRAME1 = HERE / 'data' / 'keyframe1.png'
FRAME2 = HERE / 'data' / 'keyframe2.png'

def do_request():
    files = {
        'frame1': open(FRAME1, 'rb'),
        'frame2': open(FRAME2, 'rb')
    }
    data = {'num_frames': '4', 'interpolation_mode': 'toon'}
    try:
        resp = requests.post(URL, files=files, data=data, timeout=30)
    except Exception as e:
        print('Request failed:', e)
        return
    print('Status:', resp.status_code)
    print('Headers:', resp.headers)
    # Try to mimic browser parsing
    try:
        j = resp.json()
        print('Parsed JSON:', j)
    except Exception as e:
        print('JSON parse error:', repr(e))
        print('Raw text length:', len(resp.text))
        print('Raw text (first 1000 chars):')
        print(resp.text[:1000])

if __name__ == '__main__':
    do_request()

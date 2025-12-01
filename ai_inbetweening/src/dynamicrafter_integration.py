"""
DynamiCrafter Integration shim

This module provides a safe wrapper `generate_inbetweens(frame1, frame2, num_frames, **opts)`
that attempts to call DynamiCrafter via (in order):
 - Local SDK import (hypothetical `dynamicrafter` Python package)
 - Remote HTTP API (URL from env `DYNAMICRAFTER_API_URL`)
 - Fallback to a local linear interpolation if neither is available

The function returns a list of NumPy uint8 RGB frames (H,W,3).
"""

import os
import base64
import io
from typing import List
import numpy as np
from PIL import Image


def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert('RGB'))


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype('uint8'))


def generate_inbetweens(frame1: np.ndarray, frame2: np.ndarray, num_frames: int = 4, **opts) -> List[np.ndarray]:
    """Generate inbetween frames using DynamiCrafter if available.

    Behaviour:
    - Try local SDK import `dynamicrafter` and use it if present.
    - Else if env `DYNAMICRAFTER_API_URL` is set, POST frames and expect JSON response
      with base64-encoded PNG frames under key `frames`.
    - Otherwise fall back to a simple linear interpolation.

    Returns a list of NumPy uint8 RGB frames (not including the two keyframes).
    """
    # Try local SDK
    try:
        import dynamicrafter
        try:
            print("✓ DynamiCrafter SDK found — using local SDK")
            # SDK usage is hypothetical; adapt if real SDK differs
            pil1 = _numpy_to_pil(frame1)
            pil2 = _numpy_to_pil(frame2)
            # Some SDKs accept PIL images, file paths, or numpy arrays
            result = dynamicrafter.generate(pil1, pil2, num_frames=num_frames, **opts)
            # Normalize result to list of numpy arrays
            frames = []
            for f in result:
                if isinstance(f, np.ndarray):
                    frames.append(f.astype('uint8'))
                elif isinstance(f, Image.Image):
                    frames.append(_pil_to_numpy(f))
                else:
                    # try to convert via PIL
                    frames.append(_pil_to_numpy(Image.fromarray(np.asarray(f))))
            return frames
        except Exception as e:
            print(f"⚠ DynamiCrafter SDK call failed: {e}")
    except Exception:
        # SDK not available
        pass

    # Try HTTP API
    api_url = os.environ.get('DYNAMICRAFTER_API_URL')
    if api_url:
        try:
            print(f"✓ DynamiCrafter API URL found — calling {api_url}")
            import requests
            buf1 = io.BytesIO()
            buf2 = io.BytesIO()
            Image.fromarray(frame1.astype('uint8')).save(buf1, format='PNG')
            Image.fromarray(frame2.astype('uint8')).save(buf2, format='PNG')
            buf1.seek(0)
            buf2.seek(0)
            files = {
                'frame1': ('frame1.png', buf1, 'image/png'),
                'frame2': ('frame2.png', buf2, 'image/png')
            }
            data = {'num_frames': str(num_frames)}
            timeout = int(os.environ.get('DYNAMICRAFTER_API_TIMEOUT', '60'))
            resp = requests.post(api_url, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
            j = resp.json()
            if not isinstance(j, dict):
                raise ValueError('Unexpected response format from DynamiCrafter API')
            frames_b64 = j.get('frames')
            if not frames_b64:
                raise ValueError('No "frames" returned from DynamiCrafter API')
            out = []
            for b in frames_b64:
                # b may be raw bytes or base64 string
                if isinstance(b, str):
                    b64 = b
                    data_bytes = base64.b64decode(b64)
                elif isinstance(b, (bytes, bytearray)):
                    data_bytes = bytes(b)
                else:
                    # unexpected type
                    continue
                im = Image.open(io.BytesIO(data_bytes)).convert('RGB')
                out.append(_pil_to_numpy(im))
            if out:
                return out
            else:
                raise ValueError('DynamiCrafter API returned no frames after decoding')
        except Exception as e:
            print(f"⚠ DynamiCrafter HTTP API call failed: {e}")

    # Fallback: linear interpolation
    print("⚠ DynamiCrafter not available — falling back to linear interpolation")
    f1 = frame1.astype(np.float32) / 255.0
    f2 = frame2.astype(np.float32) / 255.0
    out = []
    for i in range(1, num_frames + 1):
        t = i / (num_frames + 1)
        interp = (1 - t) * f1 + t * f2
        out.append((interp * 255).astype('uint8'))
    return out

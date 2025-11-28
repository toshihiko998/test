"""
AI Inbetweening package
"""

from .inbetweening_engine import InbetWeeningEngine
from .keyframe_loader import KeyframeLoader
from .frame_interpolation import FrameInterpolator

__version__ = '0.1.0'
__all__ = [
    'InbetWeeningEngine',
    'KeyframeLoader',
    'FrameInterpolator',
]

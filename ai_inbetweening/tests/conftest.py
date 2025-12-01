"""
Pytest configuration and shared fixtures for AI Inbetweening System tests.
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

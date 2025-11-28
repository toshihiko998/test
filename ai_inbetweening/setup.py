from setuptools import setup, find_packages

setup(
    name='ai-inbetweening',
    version='0.1.0',
    description='AI-powered anime inbetweening system',
    author='Your Name',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'Pillow>=10.0.0',
        'imageio>=2.33.0',
        'scikit-image>=0.21.0',
        'tqdm>=4.66.0',
    ],
)

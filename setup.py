import subprocess
from setuptools import setup, find_packages

# Detect CUDA availability
def has_cuda():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Set onnxruntime based on CUDA availability
onnxruntime_pkg = 'onnxruntime-gpu' if has_cuda() else 'onnxruntime'

setup(
    name='napari-organoid-counter',
    author='christinab12',
    author_email='christina.bukas@helmholtz-munich.de',
    url='https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter',
    license='MIT',
    description='A plugin to automatically count lung organoids using Deep Learning.',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'napari[all]>=0.4.17,<0.5.0',
        'napari-aicsimageio>=0.7.2',
        'torch',
        'torchvision',
        'opencv-python',
        f'{onnxruntime_pkg}>=1.23.0',
    ],
    extras_require={
        'testing': [
            'tox',
            'pytest',
            'pytest-cov',
            'pytest-qt',
            'napari',
            'pyqt5',
        ],
    },
    package_data={'': ['*.yaml']},
    entry_points={
        'napari.manifest': [
            'napari-organoid-counter = napari_organoid_counter:napari.yaml',
        ],
    },
)

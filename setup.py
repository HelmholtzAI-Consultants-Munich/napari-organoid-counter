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
    python_requires='>=3.9, <3.11',
    install_requires=[
        'napari[all]>=0.4.17,<0.5.0',
        'bioio==3.3.0',
        'bioio-ome-tiff==1.4.0',
        'bioio-tifffile==1.3.0',
        'bioio-imageio==1.3.0',
        'bioio-czi==2.6.0',
        'bioio-nd2==1.6.2',
        'bioio-lif==1.4.0',
        'bioio-dv==1.2.0',
        'torch<=2.5.1',
        'lxml_html_clean==0.4.4',
        'torchvision<=0.20.1',
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

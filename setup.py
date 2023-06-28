from setuptools import setup, find_packages

setup(
    name='audio-lid',
    version='0.1',
    py_modules=["audio_lid"],
    entry_points={
        "console_scripts": [
            "audio_lid=audio_lid:run",
        ],
    },
    description='A project that detect the language from a audio file',
    author='bevis',
    author_email='javashu2012@gmail.com',
    url='https://github.com/androidshu/audio-lid.git',
    packages=find_packages(),
    install_requires=[
        'fairseq@git+https://github.com/facebookresearch/fairseq.git@a29952ce6d313a4daf3e90647f8bf84cc6d4df6d',
        'pytest-runner==6.0.0',
        'paddlespeech',
        'paddlepaddle==2.4.1',
        'scipy',
        'tqdm',
        'Pillow',
        'torch>=1.10.0'
    ]
)

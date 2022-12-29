from setuptools import setup, find_packages
import ppasr

PPASR_VERSION = ppasr.__version__


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_requirements():
    with open('./requirements.txt', encoding="utf-8") as f:
        requirements = f.readlines()
    return requirements


if __name__ == "__main__":
    setup(
        name='ppasr',
        packages=find_packages(exclude='download_data/'),
        package_data={'': ['infer_utils/silero_vad.onnx']},
        author='yeyupiaoling',
        version=PPASR_VERSION,
        install_requires=parse_requirements(),
        description='Automatic speech recognition toolkit on PaddlePaddle',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='https://github.com/yeyupiaoling/PPASR',
        download_url='https://github.com/yeyupiaoling/PPASR.git',
        keywords=['asr', 'paddle'],
        classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Natural Language :: Chinese (Simplified)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9', 'Topic :: Utilities'
        ],
        license='Apache License 2.0',
        ext_modules=[])

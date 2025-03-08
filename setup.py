# setup.py
from setuptools import setup, find_packages

setup(
    name='naga_helper',
    version='0.2.11',
    description='Analyze reports of riichi mahjong AI NAGA and Mortal.',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    author='Jianing Shi',
    author_email='1176827825@qq.com',
    url='https://github.com/sjn4048/naga_helper',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

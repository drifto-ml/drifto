from setuptools import setup

setup(
    name='drifto',
    version='0.1.0',
    license='ELv2',
    description='Automatic featurization and ML for event analytics',
    author='Drifto Technologies Inc',
    packages=['drifto', 'drifto.ml'],
    url = 'https://github.com/drifto-ml/drifto',
    keywords = ['data', 'machine learning', 'feature engineering', 'analytics', 'autoML'],
    install_requires=['pyarrow',
                      'duckdb',
                      'torch',
                      'pytorch_lightning',
                      'onnx']
)

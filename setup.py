import setuptools

setuptools.setup(
    name='GammaTransients',
    version='1.0',
    packages=[
        'gamma_transients',
    ],
    python_requires='>=3.9',
    install_requires=[
        'dill',
        'tqdm',
        'pandas',
        'requests',
        'gammapy>=1.0',
    ],
    scripts=[
        'gamma_transients/bin/gt_scanner',
        'gamma_transients/bin/gt_plotter',
    ]
)

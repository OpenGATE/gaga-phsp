import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gaga",
    version="0.1",
    author="David Sarrut",
    author_email="david.sarrut@creatis.insa-lyon.fr",
    description="Python tools for GATE GAN simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.in2p3.fr/davidsarrut/gaga",
    package_dir={'':'src'},
    packages=setuptools.find_packages('gaga'),
    #packages=['pygan'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'tqdm',
        'json_tricks',
        'colorama',
        'pykeops',
        'geomloss',
        'seaborn'
      ],
    scripts=[
        'bin/gaga_train',
        'bin/gaga_info',
        'bin/gaga_plot',
        'bin/gaga_kl',
        'bin/gaga_wasserstein',
    ]
)

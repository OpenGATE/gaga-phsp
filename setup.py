import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gaga-phsp",
    version="0.5.2",
    author="David Sarrut",
    author_email="david.sarrut@creatis.insa-lyon.fr",
    description="Python tools for GATE GAN simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsarrut/gaga",
    packages=['gaga'],
    package_dir={'gaga': 'src'},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'tqdm',
        'colorama',
        'click',
        'scipy',
        #'torch'   # better to install torch manually to match cuda version
      ],
    scripts=[
        'bin/gaga_train',
        'bin/gaga_info',
        'bin/gaga_plot',
        'bin/gaga_generate',
        'bin/gaga_convert_pth_to_pt',
    ]
)

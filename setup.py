import setuptools
from setuptools import find_packages

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gaga-phsp",
    version="0.6.0",
    author="David Sarrut",
    author_email="david.sarrut@creatis.insa-lyon.fr",
    description="Python tools for GATE GAN simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsarrut/gaga-phsp",
    packages=find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "tqdm",
        "colorama",
        "click",
        "scipy",
        "garf",
        # 'torch'   # better to install torch manually to match cuda version
    ],
    scripts=[
        "bin/gaga_train",
        "bin/gaga_info",
        "bin/gaga_plot",
        "bin/gaga_generate",
        "bin/gaga_gauss_test",
        "bin/gaga_gauss_plot",
        "bin/gaga_convert_pth_to_pt",
        "bin/gaga_wasserstein",
        "bin/gaga_garf_generate_img",
        "bin/gaga_pairs_to_tlor",
        "bin/gaga_tlor_to_pairs",
        "bin/gaga_pet_to_pairs_old",
        "bin/gaga_pet_to_pairs",
        "bin/gaga_pet_merge_pairs",
        "bin/gaga_exit_pos_to_ideal_pos",
        "bin/gaga_ideal_pos_to_exit_pos",
    ],
)

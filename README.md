# Jaired Features Versus OpenCV SIFT Features
This repository provides a Jupyter notebook to show Jaired features compared against OpenCV's SIFT features. 

Click on [main](/main.ipynb) to see the notebook. Three images are provided for testing, but you are encouraged to use your own and experiment with the code. It uses image 8 and image 9, which are fairly similar.

An example with image 7 and image 9, which are more dissimilar, is provided in [this pdf](/match7-9.pdf).

# Installing
```bash
conda create -n conv python=3.9
conda activate conv
conda install -c anaconda jupyter
conda install Pillow numpy scipy opencv
```
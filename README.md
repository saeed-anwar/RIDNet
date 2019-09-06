# Real Image Denoising with Feature Attention
This repository is for Real Image Denoising with Feature Attention (RIDNet) introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/),  [Nick Barnes], "Real Image Denoising with Feature Attention", [[arXiv]](https://arxiv.org/abs/1904.07396) 

The model is built in PyTorch 1.1.0 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1).

Our DRLN is also available in PyTorch 0.4.0 and 0.4.1. You can download this version from [Google Drive](https://drive.google.com/open?id=1I91VGposSoFq6UrWBuxCKbst6sMi1yhR) or [here](https://icedrive.net/0/cb986Jh8rh). 

## Contents
1. [Introduction](#introduction)
2. [Network](#network)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Deep convolutional neural networks perform better on images containing spatially invariant noise (synthetic noise); however, their performance is limited on real-noisy photographs and requires multiple stage network modeling. To advance the practicability of denoising algorithms, this paper proposes a novel single-stage blind real image denoising network (RIDNet) by employing a modular architecture. We use a residual on the residual structure to ease the flow of low-frequency information and apply feature attention to exploit the channel dependencies. Furthermore, the evaluation in terms of quantitative metrics and visual quality on three synthetic and four real noisy datasets against 19 state-of-the-art algorithms demonstrate the superiority of our RIDNet.


<img align="centre" img src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/Front.PNG" width="600">
Sample results on a real noisy face image from RNI15 dataset.

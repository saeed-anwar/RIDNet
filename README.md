# Real Image Denoising with Feature Attention
This repository is for Real Image Denoising with Feature Attention (RIDNet) introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/),  [Nick Barnes], "Real Image Denoising with Feature Attention", [[arXiv]](https://arxiv.org/abs/1904.07396) 

The model is built in PyTorch 0.4.0, PyTorch 0.4.1 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1).


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

<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/Front.PNG">
</p>
Sample results on a real noisy face image from RNI15 dataset.

## Network
![Network](/Figs/Net.PNG)
The architecture of the proposed network. Different green colors of the conv layers denote different dilations while the smaller
size of the conv layer means the kernel is 1x1. The second row shows the architecture of each EAM.

<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/RIDNet/blob/master/Figs/Front.PNG">
</p>
The feature attention mechanism for selecting the essential features.

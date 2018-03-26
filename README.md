# tps_stn_pytorch
PyTorch implementation of Spatial Transformer Network (STN) with Thin Plate Spline (TPS).

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/top_1.gif" height = "200"/>

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/top_2.gif" height = "200"/>

## Introduction

STN is a powerful neural network architecture proposed by DeepMind in [[1]](#ref-1).  STN achieves real spatial invariance by automatically rectify input images before they are feeded into a normal classification network. The most amazing part of STN is that it is end-to-end differential and can be directly plugged into existing network architectures (AlexNet, Resnet, etc), **without any extra supervision.**

Original STN paper [[1]](#ref-1) experiments on three specific transformation forms: Affine Transformation, Projective Transformation and **Thin Plate Spline Transformation (TPS)**.  Among them I think TPS is the most powerful translation because it can warp a image in arbitrary way. As shown below, I can warp my Avatar

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/source_avatar.jpg" height = "200"/>

into

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/target_avatar.jpg" height = "200"/>

TPS-STN has been used in OCR application [[2]](#ref-2). In this paper TPS-STN is to automatically rectify distorted text images, before they are feeded into a normal OCR text recognition model:

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/ocr.jpg" height = "130"/>

## Dependencies

- Python3
- PyTorch
- Torchvision
- Numpy
- Matplotlib
- Pillow / PIL
- imageio

I use `imageio` to create GIF visualization. Simply install it by `pip install imageio`.

## Run

	python mnist_train.py --model unbounded_stn --angle 90 --grid_size 4
	python mnist_visualize.py --model unbounded_stn --angle 90 --grid_size 4
	python mnist_make_gif.py --model unbounded_stn --angle 90 --grid_size 4

Then PNG and GIF resutls will be saved in `./image/unbounded_stn_angle60_grid4/` and `./gif/unbounded_stn_angle60_grid4/`.

You can try other combinations of model architecture, mnist random rotation angle and TPS grid size. Details below.

## Arguments

There are three controllable arguments: `--model`, `--angle`, `--grid_size`.

`--model`: str, required
- With `no_stn`, STN module is discarded and only a single CNN classifier remains.
- With `bounded_stn`, the output of localization network is squeezed to [-1, 1] by `F.tanh`, as was done in [[2]](#ref-1)
- With `unbounded_stn`, the output of locolizaition network is not squeezed

`--angle`: int, default = 60
- Samples in MNIST dataset will be rotated by random angles within `[-angle, angle]`

`--grid_size`: int, default = 4
- Use `(grid_size x grid_size)` control points to define Thin Plate Spline transformation

## Test Accuracy Curves

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/curve.png" height = "900"/>

## Visualize TPS-STN

Results with `angle = 90` are generally bad:

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/angle90_bad1.gif" height = "200"/>

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/angle90_bad2.gif" height = "200"/>

Results with `bounded_stn` are bad if `grid_size <= 3`:

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/bounded_bad1.gif" height = "200"/>

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/bounded_bad2.gif" height = "200"/>

But ok if `grid_size >= 4`:

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/bounded_good1.gif" height = "200"/>

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/bounded_good2.gif" height = "200"/>

Results with `unbounded_stn` are ok:

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/unbounded_good1.gif" height = "200"/>

<img src="https://raw.githubusercontent.com/warbean/tps_stn_pytorch/master/demo/unbounded_good2.gif" height = "200"/>

Of course there are always bad cases in each combination. You can download all my GIFs from [Baidu Netdisk](https://pan.baidu.com/s/1c2KXPw0) (file size 2G).



## Reference

[1] [Spatial Transformer Networks](https://arxiv.org/abs/1509.05329)<span id="ref-1"/>

[2] [Robust Scene Text Recognition with Automatic Rectiﬁcation](https://arxiv.org/abs/1603.03915)<span id="ref-2"/>

[3] [数值方法——薄板样条插值（Thin-Plate Spline）](http://blog.csdn.net/VictoriaW/article/details/70161180)<span id="ref-3"/>

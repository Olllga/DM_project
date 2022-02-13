# GAN: generating brushstroke paintings

This is the repository of the project proposed for the course of *Data Mining*.

This work is centered on the idea of **Intrinsic Style Transfer** from the paper by [Reiichiro Nakano, **"Neural Painters: A learned differentiable constraint for generating brushstroke paintings"** (2019)](https://arxiv.org/abs/1904.08410) and [**MyPaintBrushstrokes**](https://www.kaggle.com/reiinakano/mypaint_brushstrokes) dataset.

## Introduction

The core idea behind the Neural Painters technology is to provide **the painter agent** trained to generate the source image as a painting in a stroke-by-stroke fashion as a human artist would do.

```Input: source image.```

```Output: painting of this image by brushstrokes```.

The following **4 source images** have been borrowed from the repository of the original paper and preprocessed to be for testing of the built models and experimenting with the results:

<p align="center">
  <img src="/images/source_images/vanc.jpg" width="247" height="151"/><img src="/images/source_images/louvre.png" width="190" height="151"/> <img src="/images/source_images/eiffel.jpeg" width="151" height="151"/> <img src="/images/source_images/castle.png" width="151" height="151"/> 
</p>

### Dataset
Dataset is not stored in the project repository due to its large memory consumption, but can be downloaded from the corresponding Kaggle web-page [**MyPaintBrushstrokes**](https://www.kaggle.com/reiinakano/mypaint_brushstrokes) as:

```!kaggle datasets download -d reiinakano/mypaint_brushstrokes```

It is structured in the following way:

|        Name         | N samples | File size |
| -----------------   |:---------:|:---------:|
|```episodes_0.npz``` |   100000  |  196.3MB  |
|        ...          |    ...    |    ...    |
|```episodes_77.npz```|   100000  |  196.3MB  |

```episodes_0.npz```, ```episodes_1.npz```, ```episodes_2.npz``` were mainly used in the project.


Each data sample is represented by **action** and the **stroke**. 
From the semantical point of view given action realizes given stroke, and

- **action:** array of length ```12```,
- **stroke:** RGB image with ```(img_width, img_height) = (64, 64)```.

Examples of the stroke images are printed below:

<p align="center">
  <img src="/images/readme_images/dataset_examples.png" width="640" height="320"/>
</p>

## Author
- Olga Sorokoletova - 1937430

<p align="center">
    <img src="assets/keras_gan.png" width="480"\>
</p>

## Keras-GAN
Collection of Keras implementations of Generative Adversarial Networks (GANs) suggested in research papers. These models are in some cases simplified versions of the ones ultimately described in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GAN varieties to implement are very welcomed.

<b>See also:</b> [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#ac-gan)
    + [Adversarial Autoencoder](#adversarial-autoencoder)
    + [Bidirectional GAN](#bigan)
    + [Boundary-Seeking GAN](#bgan)
    + [Conditional GAN](#cgan)
    + [Context-Conditional GAN](#cc-gan)
    + [Context Encoder](#context-encoder)
    + [Coupled GANs](#cogan)
    + [CycleGAN](#cyclegan)
    + [Deep Convolutional GAN](#dcgan)
    + [DiscoGAN](#discogan)
    + [DualGAN](#dualgan)
    + [Generative Adversarial Network](#gan)
    + [InfoGAN](#infogan)
    + [LSGAN](#lsgan)
    + [Pix2Pix](#pix2pix)
    + [PixelDA](#pixelda)
    + [Semi-Supervised GAN](#sgan)
    + [Super-Resolution GAN](#srgan)
    + [Wasserstein GAN](#wgan)
    + [Wasserstein GAN GP](#wgan-gp)     

## Installation
    $ git clone https://github.com/eriklindernoren/Keras-GAN
    $ cd Keras-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   
### AC-GAN
- Discriminatorに「多クラス分類を追加」

- よりバリエーションの多い画像出力を可能とする手法

Implementation of _Auxiliary Classifier Generative Adversarial Network_.

[Code](acgan/acgan.py)

Paper: https://arxiv.org/abs/1610.09585

#### Example
```
$ cd acgan/
$ python3 acgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/acgan.gif" width="640"\>
</p>

### Adversarial Autoencoder

- オートエンコーダの潜在ベクトル を任意の分布かどうかを騙す
- Discreminatorは潜在ベクトルか任意の分布かを判定する

Implementation of _Adversarial Autoencoder_.

[Code](aae/aae.py)

Paper: https://arxiv.org/abs/1511.05644

#### Example
```
$ cd aae/
$ python3 aae.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/aae.png" width="640"\>
</p>

### BiGAN

- データxを潜在空間にマッピングするエンコーダを追加
- generatorは潜在空間の値を用いてデータを生成

Implementation of _Bidirectional Generative Adversarial Network_.

[Code](bigan/bigan.py)

Paper: https://arxiv.org/abs/1605.09782

#### Example
```
$ cd bigan/
$ python3 bigan.py
```

### BGAN

- 離散データでGANを訓練

Implementation of _Boundary-Seeking Generative Adversarial Networks_.

[Code](bgan/bgan.py)

Paper: https://arxiv.org/abs/1702.08431

#### Example
```
$ cd bgan/
$ python3 bgan.py
```

### CC-GAN

- 画像復元（四角い穴を埋める）

Implementation of _Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks_.

[Code](ccgan/ccgan.py)

Paper: https://arxiv.org/abs/1611.06430

#### Example
```
$ cd ccgan/
$ python3 ccgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/ccgan.png" width="640"\>
</p>

### CGAN

- Generatorの入力に条件ベクトル(カテゴリ)を加えた

Implementation of _Conditional Generative Adversarial Nets_.

[Code](cgan/cgan.py)

Paper:https://arxiv.org/abs/1411.1784

#### Example
```
$ cd cgan/
$ python3 cgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/cgan.gif" width="640"\>
</p>

### Context Encoder

- 画像復元（四角い穴を埋める）
- 名の通りencoderを追加している

Implementation of _Context Encoders: Feature Learning by Inpainting_.

[Code](context_encoder/context_encoder.py)

Paper: https://arxiv.org/abs/1604.07379

#### Example
```
$ cd context_encoder/
$ python3 context_encoder.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/context_encoder.png" width="640"\>
</p>

### CoGAN

- 同一表現ベクトルzから2つのgeneratorで別ドメインの画像を生成
- Disciminatorが生成画像とドメイン実画像を識別

Implementation of _Coupled generative adversarial networks_.

[Code](cogan/cogan.py)

Paper: https://arxiv.org/abs/1606.07536

#### Example
```
$ cd cogan/
$ python3 cogan.py
```

### CycleGAN

- ソースドメインXからターゲットドメインYへの画像の変換

  

Implementation of _Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_.

[Code](cyclegan/cyclegan.py)

Paper: https://arxiv.org/abs/1703.10593

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan.png" width="640"\>
</p>

#### Example
```
$ cd cyclegan/
$ bash download_dataset.sh apple2orange
$ python3 cyclegan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan_gif.gif" width="640"\>
</p>

### DCGAN

- GeneratorとDiscriminatorのそれぞれのネットワークに畳み込み層を使用している

Implementation of _Deep Convolutional Generative Adversarial Network_.

[Code](dcgan/dcgan.py)

Paper: https://arxiv.org/abs/1511.06434

#### Example
```
$ cd dcgan/
$ python3 dcgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/dcgan2.png" width="640"\>
</p>

### DiscoGAN

- 2つのラベル無し画像の集合を用いた学習
- 事前学習が不要な学習モデル
- 双方向にドメイン変換が可能になっている

Implementation of _Learning to Discover Cross-Domain Relations with Generative Adversarial Networks_.

[Code](discogan/discogan.py)

Paper: https://arxiv.org/abs/1703.05192

<p align="center">
    <img src="http://eriklindernoren.se/images/discogan_architecture.png" width="640"\>
</p>

#### Example
```
$ cd discogan/
$ bash download_dataset.sh edges2shoes
$ python3 discogan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/discogan.png" width="640"\>
</p>

### DualGAN

-  domain 変換(例えばイラスト↔ 写真といった変換)
- イラストの集合と写真の集合」がデータとして与えられている状況

Implementation of _DualGAN: Unsupervised Dual Learning for Image-to-Image Translation_.

[Code](dualgan/dualgan.py)

Paper: https://arxiv.org/abs/1704.02510

#### Example
```
$ cd dualgan/
$ python3 dualgan.py
```

### GAN
Implementation of _Generative Adversarial Network_ with a MLP generator and discriminator.

[Code](gan/gan.py)

Paper: https://arxiv.org/abs/1406.2661

#### Example
```
$ cd gan/
$ python3 gan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/gan_mnist5.gif" width="640"\>
</p>

### InfoGAN

- generatorの入力をc（latent variable）とする
- CGANとの違いは暗黙的に生成データxとcを結びつける
- （CGANは明示的に生成データxとcを結びつける）

Implementation of _InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets_.

[Code](infogan/infogan.py)

Paper: https://arxiv.org/abs/1606.03657

#### Example
```
$ cd infogan/
$ python3 infogan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/infogan.png" width="640"\>
</p>

### LSGAN

- 二乗誤差を使うことで普通のDCGANより本物に近い画像
- 実装が簡単

Implementation of _Least Squares Generative Adversarial Networks_.

[Code](lsgan/lsgan.py)

Paper: https://arxiv.org/abs/1611.04076

#### Example
```
$ cd lsgan/
$ python3 lsgan.py
```

### Pix2Pix

- 2つのペアの画像から画像間の関係を学習
- 1枚の画像からその関係を考慮した補間をしてペアの画像を生成する

Implementation of _Image-to-Image Translation with Conditional Adversarial Networks_.

[Code](pix2pix/pix2pix.py)

Paper: https://arxiv.org/abs/1611.07004

<p align="center">
    <img src="http://eriklindernoren.se/images/pix2pix_architecture.png" width="640"\>
</p>

#### Example
```
$ cd pix2pix/
$ bash download_dataset.sh facades
$ python3 pix2pix.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/pix2pix2.png" width="640"\>
</p>

### PixelDA

- Domain Adaptation

- Domain A及びDomain Bから共通的な特徴を抽出する

Implementation of _Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks_.

[Code](pixelda/pixelda.py)

Paper: https://arxiv.org/abs/1612.05424

#### MNIST to MNIST-M Classification
Trains a classifier on MNIST images that are translated to resemble MNIST-M (by performing unsupervised image-to-image domain adaptation). This model is compared to the naive solution of training a classifier on MNIST and evaluating it on MNIST-M. The naive model manages a 55% classification accuracy on MNIST-M while the one trained during domain adaptation gets a 95% classification accuracy.

```
$ cd pixelda/
$ python3 pixelda.py
```

| Method       | Accuracy  |
| ------------ |:---------:|
| Naive        | 55%       |
| PixelDA      | 95%       |

### SGAN
Implementation of _Semi-Supervised Generative Adversarial Network_.

[Code](sgan/sgan.py)

Paper: https://arxiv.org/abs/1606.01583

#### Example
```
$ cd sgan/
$ python3 sgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/sgan.png" width="640"\>
</p>

### SRGAN

- 低画質 to 高画質

Implementation of _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_.

[Code](srgan/srgan.py)

Paper: https://arxiv.org/abs/1609.04802

<p align="center">
    <img src="http://eriklindernoren.se/images/superresgan.png" width="640"\>
</p>


#### Example
```
$ cd srgan/
<follow steps at the top of srgan.py>
$ python3 srgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/srgan.png" width="640"\>
</p>

### WGAN

- Wasserstein距離により損失関数を設計

Implementation of _Wasserstein GAN_ (with DCGAN generator and discriminator).

[Code](wgan/wgan.py)

Paper: https://arxiv.org/abs/1701.07875

#### Example
```
$ cd wgan/
$ python3 wgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/wgan2.png" width="640"\>
</p>

### WGAN GP

- 学習の安定化、計算方法の工夫

- Grgdient penalityを導入

Implementation of _Improved Training of Wasserstein GANs_.

[Code](wgan_gp/wgan_gp.py)

Paper: https://arxiv.org/abs/1704.00028

#### Example
```
$ cd wgan_gp/
$ python3 wgan_gp.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/imp_wgan.gif" width="640"\>
</p>

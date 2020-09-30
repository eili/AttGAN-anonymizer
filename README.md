# Face anonymization using AttGAN-PyTorch

The anonymization method uses AttGAN, an encoder-decoder based neural network, for changing facial attributes. Anonymization is achieved by applying changes to different facial characteristics in two stages. Experiments show promising results, with up to 100% face detection rate. The new method achieves a high face detection rate by maintaining a human-looking face. However, optimizing the anonymization process requires individual adaptions. What works well for one face may not work well for another.

Anonymization is the process of concealing the identity of persons in the data. The amount of change applied by the anonymization method can be represented numerically as a distance between the original and anonymized face. The distance is calculated using facenet-pytorch.

![Teaser](https://github.com/eili/AttGAN-anonymizer/blob/master/pics/5941_test1.jpg)
Test on the CelebA identity 5941
![Teaser](https://github.com/eili/AttGAN-anonymizer/blob/master/pics/6011_test1.jpg)
Test on the CelebA identity 6011
![Teaser](https://github.com/eili/AttGAN-anonymizer/blob/master/pics/9739_test1.jpg)
Test on the CelebA identity 9739
![Teaser](https://github.com/eili/AttGAN-anonymizer/blob/master/pics/10154_test1.jpg)
Test on the CelebA identity 10154
![Teaser](https://github.com/eili/AttGAN-anonymizer/blob/master/pics/n00284_1.jpg)
Test on the VGGFaces2 n00284 image
![Teaser](https://github.com/eili/AttGAN-anonymizer/blob/master/pics/n00284_2.jpg)
Another test on the VGGFaces2 n00284 image

Based on the AttGAN-PyTorch:
First install AttGAN-PyTorch and make it run.
The anonymization process generates a set of random attributes for the AttGAN. Image is processed two times with these attributes, the second time
the attributes are inverted.


A PyTorch implementation of AttGAN - [Arbitrary Facial Attribute Editing: Only Change What You Want](https://arxiv.org/abs/1711.10678)

![Teaser](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/teaser.jpg)
Test on the CelebA validating set

![Custom](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/custom_images.jpg)
Test on my custom set

Inverting 13 attributes respectively. From left to right: _Input, Reconstruction, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Male, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Young_

The original TensorFlow version can be found [here](https://github.com/LynnHo/AttGAN-Tensorflow).


## Requirements

* Python 3.6
* PyTorch 1.3.1
* torchvision 0.4.1
* tensorboardX
* pandas 0.25.3
* pillow 6.2.1
* matplotlib 3.1.1
* numpy 1.17.4
* scikit-learn 0.22.2
* scipy 1.3.2

* facenet-pytorch 1.0.1

```bash
pip3 install -r requirements.txt
```

If you'd like to train with __multiple GPUs__, please install PyTorch __v0.4.0__ instead of v1.0.0 or above. The so-called stable version of PyTorch has a bunch of problems with regard to `nn.DataParallel()`. E.g. https://github.com/pytorch/pytorch/issues/15716, https://github.com/pytorch/pytorch/issues/16532, etc.

```bash
pip3 install --upgrade torch==0.4.0
```

* Dataset
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
    * [Images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip) should be placed in `./data/img_align_celeba/*.jpg`
    * [Attribute labels](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt) should be placed in `./data/list_attr_celeba.txt`
  * [HD-CelebA](https://github.com/LynnHo/HD-CelebA-Cropper) (optional)
    * Please see [here](https://github.com/LynnHo/HD-CelebA-Cropper).
  * [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset (optional)
    * Please see [here](https://github.com/willylulu/celeba-hq-modified).
    * _Images_ should be placed in `./data/celeba-hq/celeba-*/*.jpg`
    * _Image list_ should be placed in `./data/image_list.txt`
* [Pretrained models](http://bit.ly/attgan-pretrain): download the models you need and unzip the files to `./output/` as below,
  ```text
  output
  ├── 128_shortcut1_inject0_none
  ├── 128_shortcut1_inject1_none
  ├── 256_shortcut1_inject0_none
  ├── 256_shortcut1_inject1_none
  ├── 256_shortcut1_inject0_none_hq
  ├── 256_shortcut1_inject1_none_hq
  ├── 384_shortcut1_inject0_none_hq
  └── 384_shortcut1_inject1_none_hq
  ```
See original's readme.

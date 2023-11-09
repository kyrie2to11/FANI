# Feature Aggregating Network with Inter-Frame Interaction for Efficient Video Super-Resolution

## Overview

This repository provides the tensorflow implementation of the Video Super-resolution model **FANI** for mobile devices.

### Contents

- [Feature Aggregating Network with Inter-Frame Interaction for Efficient Video Super-Resolution](#feature-aggregating-network-with-inter-frame-interaction-for-efficient-video-super-resolution)
  - [Overview](#overview)
    - [Contents](#contents)
    - [Requirements](#requirements)
    - [Dataset preparation](#dataset-preparation)
    - [Training and Validation](#training-and-validation)
      - [Configuration](#configuration)
      - [Training](#training)
      - [Validation](#validation)
    - [Testing](#testing)
    - [Convert to tflite](#convert-to-tflite)
    - [TFLite inference on Mobile](#tflite-inference-on-mobile)
    - [License](#license)
    - [Citation](#citation)

---

### Requirements

- Python: 3.8.10
- Python packages: numpy, imageio and pyyaml 
- [TensorFlow = 2.11.0](https://www.tensorflow.org/install/) + [CUDA cuDNN](https://developer.nvidia.com/cudnn)
- GPU for training (e.g., Nvidia GeForce RTX 3090)

[[back]](#contents)

---

### Dataset preparation

- Download REDS dataset and extract it into `data` folder.

  <sub>The REDS dataset folder should contain three subfolders: `train/`, `val/` and `test/`. </sub>
  <sub>Please find the download links to above files in [website](https://seungjunnah.github.io/Datasets/reds.html). </sub>

[[back]](#contents)

---

### Training and Validation

#### Configuration

Before training and testing, please make sure the fields in `config.yml` is properly set.

```yaml
log_dir: snapshot -> The directory which records logs and checkpoints. 

dataset:
    dataloader_settings: -> The setting of different splits dataloader.
        train:
            batch_size: 4
            drop_remainder: True
            shuffle: True
            num_parallel_calls: 6
        val:
            batch_size: 1
    data_dir: data/ -> The directory of REDS dataset.
    degradation: sharp_bicubic -> The degradation of images.
    train_frame_num: 10 -> The number of image frame(s) for per training step.
    test_frame_num: 100 -> The number of image frame(s) for per testing step.
    crop_size: 64 -> The height and width of cropped patch.

model:
    path: model/mobile_rrn.py -> The path of model file.
    name: MobileRRN -> The name of model class.

learner:
    general:
        total_steps: 1500000 -> The number of training steps.
        log_train_info_steps: 100 -> The frequency of logging training info.
        keep_ckpt_steps: 10000 -> The frequency of saving checkpoint.
        valid_steps: 100000 -> The frequency of validation.

    optimizer: -> Define the module name and setting of optimizer
        name: Adam
        beta_1: 0.9
        beta_2: 0.999

    lr_scheduler: -> Define the module name and setting of learning rate scheduler
        name: ExponentialDecay
        initial_learning_rate: 0.0001
        decay_steps: 1000000
        decay_rate: 0.1
        staircase: True

    saver:
        restore_ckpt: null -> The path to checkpoint where would be restored from.
```

#### Training

To train the model, use the following command:

```bash
sh train.sh
```

The main arguments are as follows:

>```process``` : &nbsp; Process type should be train or test.<br/>
>```config_path``` : &nbsp; Path of yml config file of the application.<br/>

After training, the checkpoints will be produced in `log_dir`.

#### Validation

To valid the model, use the following command:

```bash
sh valid.sh
```

After validating, the output images will be produced in `log_dir/output`.

[[back]](#contents)

---

### Testing

To generate testing outputs, use the following command:

```bash
sh test.sh
```

The main arguments are as follows:

>```model_path``` : &nbsp; Path of model file.<br/>
>```model_name``` : &nbsp; Name of model class.<br/>
>```ckpt_path``` : &nbsp; Path of checkpoint.<br/>
>```data_dir``` : &nbsp; Directory of testing frames in REDS dataset.<br/>
>```output_dir``` : &nbsp; Directory for saving output super-resolution images.<br/>

[[back]](#contents)

---

### Convert to tflite

To convert the keras model to tflite, use the following command:

```bash
sh convert_to_tflite.sh
```

The main arguments are as follows:

>```model_path``` : &nbsp; Path of model file.<br/>
>```model_name``` : &nbsp; Name of model class.<br/>
>```input_shape``` : &nbsp; Series of the input shapes split by \`:\`.<br/>
>```ckpt_path``` : &nbsp; Path of checkpoint.<br/>
>```output_tflite``` : &nbsp; Path of output tflite.<br/>

[[back]](#contents)

---

### TFLite inference on Mobile

- [AI benchmark](https://ai-benchmark.com/): An app allowing you to load your model and run it locally on your own Android devices with various acceleration options (e.g. CPU, GPU, APU, etc.).

[[back]](#contents)

---

### License

FANI is released under the [MIT license](LICENSE).

[[back]](#contents)

---
### Citation
```
@misc{https://doi.org/10.48550/arxiv.2302.12288,
  doi = {10.48550/ARXIV.2302.12288},
  
  url = {https://arxiv.org/abs/2302.12288},
  
  author = {Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
[[back]](#contents)

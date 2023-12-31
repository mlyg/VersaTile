# VersaTile

A readily-adaptable, morphology-independent, deep learning-based framework for instance segmentation designed for digital pathology.

## Source

ArXiv paper to be released soon.

## Overview

**1. What is VersaTile?** \
VersaTile is an automated pipeline designed to segment structures on digital pathology. Specifically, VersaTile produces instance segmentation labels, meaning that it is capable of distinguishing between objects of the same type.

**2. What is concept behind VersaTile?** \
As the name suggests, VersaTile is designed to be flexible:
- **Handles stain diversity**: uses our developed stain consistency learning approach to train models robust to stain variation
- **Handles morphology diversity**: uses object skeletons as a reference point to identify structures
- **Highly customisable**: plug-and-play design makes it easy to modify any part of the pipeline e.g. architecture, loss function and distance transform map

**3. How does VersaTile work?** \
VersaTile can be divided into three steps: 
1. **Semantic segmentation and distance transform map prediction**: deep neural network is trained to generate both outputs. 
2. **Vector field integration**: pixels within an object move towards the object skeleton. The result is that the distance between pixels belonging to the same object is minimised and the distance between pixels belonging to different objects is maximised. 
3. **Clustering**: pixels that cluster together after vector field integration are grouped together and labeled as a distinct object.

![Overview of loss function inheritance](https://github.com/mlyg/VersaTile/blob/main/figures/overview.png)

**4. Who is VersaTile designed for?** \
VersaTile is designed for:
1. **Users**: people who want to use VersaTile to segment their own data.
2. **Researchers**: people who want to build on VersaTile and develop new methods.

**5. How can I use VersaTile?** \
This repository contains the code necessary to train and run inference using VersaTile. Tutorial notebooks will be developed to guide new users on how to use and modify VersaTile.

Important points to note:
1. Segmentation models are from https://github.com/qubvel/segmentation_models.pytorch. The source code requires minor editing to add a second output to the models for the distance transform map. The modified code is available in 'smp_edited'.
2. The code was developed to work for the CoNIC challange data. This data stores all images in a large numpy array (rather than in a folder with separate images). If you want to use a custom dataset, then a new PyTorch Dataset class will need to be created. We plan to add more flexibility with custom datasets in the future.

## Training using VersaTile

- The code to train VersaTile is located in train.py. 
- To run train.py, a configuration file is required. 
- You can edit the configuration settings in train_config.yaml.

```
python train.py -c /path/to/train_config.yaml
```
## Inference using VersaTile

- The code to run inference using VersaTile is located in test.py
- To run test.py, a configuration file is required.
- You can edit the configuration settings in test_config.yaml

```
python test.py -c /path/to/test_config.yaml
```
## Whole-slide imaging using VersaTile
- The code to process whole-slide imaging data is located in process_wsi.py
- This code is still under development.
- To run process_wsi.py, a configuration file is required.
- You can edit the configuration settings in wsi_config.yaml

```
python process_wsi.py -c /path/to/wsi_config.yaml
```

## Acknowledgements
VersaTile is built on the work from many excellent researchers who have generously open-sourced their code. Please check out their work:
1. **Vector field integration**: https://github.com/ryanirl/torchvf/tree/main
2. **Segmentation models**: https://github.com/qubvel/segmentation_models.pytorch
3. **Gradient map**: https://github.com/vqdang/hover_net
4. **Bayesian hyperparameter optimisation**: https://github.com/hyperopt/hyperopt
5. **Metrics**: https://github.com/stardist/stardist

## License
Distributed under the Apache-2.0 license. \
Please be wary of the various licenses associated with the code in VersaTile that comes from other libraries.

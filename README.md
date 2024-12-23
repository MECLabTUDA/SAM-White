# Exploring SAM Ablations for Enhancing Medical Segmentation

This repository represents the official PyTorch code base for our paper **Exploring SAM Ablations for Enhancing Medical Segmentation**. For more details, please refer to [our paper](https://arxiv.org/pdf/2310.00504).


## Table Of Contents

1. [Installation](#installation)
3. [How to get started?](#how-to-get-started)
4. [Data and pre-trained models](#data-and-pre-trained-models)
5. [Citations](#citations)
6. [License](#license)

## Installation

The simplest way to install all dependencies is by using [Anaconda](https://conda.io/projects/conda/en/latest/index.html):

1. Create a Python 3.9 environment as `conda create -n <your_conda_env> python=3.9` and activate it as `conda activate  <your_conda_env>`.
2. Install CUDA and PyTorch through conda with the command specified by [PyTorch](https://pytorch.org/). The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`. Our code was last tested with version 1.13. Pytorch and TorchVision versions can be specified during the installation as `conda install pytorch==<X.X.X> torchvision==<X.X.X> cudatoolkit=<X.X> -c pytorch`. Note that the cudatoolkit version should be of the same major version as the CUDA version installed on the machine, e.g. when using CUDA 11.x one should install a cudatoolkit 11.x version, but not a cudatoolkit 10.x version.
3. Navigate to the project root (where `setup.py` lives).
4. Execute `pip install -r requirements.txt` to install all required packages.


## How to get started?
- The easiest way to start is using our `train_abstract_*.py` python files. For every baseline, we provide specific `train_abstract_*.py` python files, located in the [scripts folder](https://github.com/MECLabTUDA/SAM-White/tree/main/scripts).
- The [eval folder](https://github.com/MECLabTUDA/SAM-White/tree/main/eval) contains several jupyter notebooks that were used to calculate performance metrics and plots used in our submission.


## Data and pre-trained models
- **Data**: In our paper, we used four publicly available datasets from:
  - [Multimodal Brain Tumor Segmentation Challenge 2020](https://www.med.upenn.edu/cbica/brats2020/data.html)
  - [Breast Cancer Semantic Segmentation](https://bcsegmentation.grand-challenge.org/)
  - [Kvasir SEG](https://datasets.simula.no/kvasir-seg/)
  - [Cataract Dataset for Image Segmentation](https://cataracts.grand-challenge.org/CaDIS/)
- **Models**: Our pre-trained models from our submission can be provided by contacting the [main author](mailto:amin.ranem@tu-darmstadt.de) upon request.

For more information about our experiments and ablations, please read the following paper:
```
Ranem, A., Babendererde, N., Frisch Y., Krumb, H. J., Fuchs, M., & Mukhopadhyay, A. (2023).
Exploring sam ablations for enhancing medical segmentation in radiology and pathology. arXiv preprint arXiv:2310.00504.
```

## Citations
If you are using our work or code base for your article, please cite the following paper:
```
@article{ranem2023exploring,
  title={Exploring sam ablations for enhancing medical segmentation in radiology and pathology},
  author={Ranem, Amin and Babendererde, Niklas and Frisch, Yannik and Krumb, Henry John
          and Fuchs, Moritz and Mukhopadhyay, Anirban},
  journal={arXiv preprint arXiv:2310.00504},
  year={2023}
}

```

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)

# AMINN
Code for our MICCAI 2021 paper "AMINN: Autoencoder-based Multiple Instance Neural Network Improves Outcome Prediction of Multifocal Liver Metastases".

by Jianan Chen (chenjn2010@gmail.com), Helen M. C. Cheung, Laurent Milot and Anne L. Martel (anne.martel@sunnybrook.ca).

## Introduction
AMINN refers to autoencoder-based multiple instance neural network that is deveopled to address multifocality, i.e. to incorporate features from all lesions for prediction/classification:
* We jointly train an autoencoder to reconstruct input features and a multiple instance network to make predictionsby aggregating information from all tumour lesions of a patient
* We incorporate a two-step normalization technique to improve the trainingof deep neural networks, built on the observation that the distributionsof radiomic features are almost always severely skewed.
* Experimental results empirically validated our hypothesis that incorporating imaging features of all lesions improves outcome prediction for multifocal cancer.
* Our code is written in Keras, using Tensorflow as backend. If using radiomic features as input, one run of training takes few minutes on a moderate GPU or CPU.

The paper has been early accepted by MICCAI 2021. For more details, please refer to our paper. [Link](https://arxiv.org/abs/2012.06875)

## How to use
`main.py`: Trains a AMINN model. The performance of the model along with hyperparameters used to train the model are saved in a output csv file.

`myargs.py`: Defines arguments used for training the model, including model related parameters, training parameters and system parameters.

`network.py`: Contains code for training and testing. Defines the AMINN model.

`pooling_method.py` and `layer.py`: Defines max, log-sum-exponential (lse), average pooling and attention-based pooling, respectively.

`dataset.py` and `utils.py`: Contains scripts that are used for creating datasets for AMINN from csv files (radimoics features + clinical variables).

## Citing AMINN

If you find AMINN useful in your research, please consider citing:

    @article{chen2020aminn,
    title={AMINN: Autoencoder-based Multiple Instance Neural Network for Outcome Prediction of Multifocal Liver Metastases},
    author={Chen, Jianan and Cheung, Helen and Milot, Laurent and Martel, Anne L},
    journal={arXiv preprint arXiv:2012.06875},
    year={2020}
    }

## Contact
If you have any questions about this code, I am happy to answer your issues or emails (to chenjn2010@gmail.com).

## Train
```Shell
    python main.py --pooling=ave --epoch=100 --runs=10 --folds=3 --lr=1e-4 --recon_coef=1.0 --fp_coef=1.0 
```

## Acknowledgements
The work conducted by [Jianan Chen](https://gjiananchen.github.io/) was funded by Grants from the [Martel lab](https://github.com/martellab-sri).

I would also like to thank the following repositories for supporting and inspiring this work. 
* https://github.com/yanyongluan/MINNs
* https://github.com/AMLab-Amsterdam/AttentionDeepMIL
* https://github.com/utayao/Atten_Deep_MIL


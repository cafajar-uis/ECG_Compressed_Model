# Compressed Deep Learning Models to detect Cardiac Arrhythmias

ECG interpretation by deep learning models has proven to offer an important alternative in the clinical ECG workflow. However, these deep learning models are both computational and memory intensive, making them difficult to deploy on wearable devices because of their limitations on computational resources.
A small model with appropriate accuracy will be more feasible to implement in a wearable device. We propose several models with state-of-the-art accuracy and F1 scores. The proposed models were evaluated on the Incentia11K dataset. Our CNN4 baseline model with only 11,833 parameters achieved an 0.9 F1-score higher than the average F1-score of a cardiologist (0.78). The model predicts cardiac arrhythmias, normal rhythms, or noise signals from direct ECG signals without pre-processing.

## Dataset description

We chose the Icentia11K dataset [[1]](#1), the largest public ECG dataset of continuous raw signals. It contains signals from 11,000 patients and has around 2 billion labeled beats. Our specific version has 2.5 million signals. Please refer to [[1]](#1) and [[2]](#2) for more information about downloading and using this dataset. We split the data into training, validation, and test sets with 64%, 16%, and 20% of the data, respectively.

## Baseline and Compact Models 

We created and trained several models variating the number of residual blocks, the number of initial filters, and the number of outputs of each residual block. These models are better explained in our paper [[3]](#3). In the training folder, you can find a python notebook where is described in detail the process of creating and training the models. The following table shows how our five models were created.

| Models 	| Residual Blocks 	| Initial Filters 	| s_j 	| Parameters 	|
|:------:	|:---------------:	|:---------------:	|:---:	|:----------:	|
|  CNN1  	|        8        	|        2        	|  8  	|    4,455   	|
|  CNN2  	|        13       	|        2        	|  8  	|    7,093   	|
|  CNN3  	|        11       	|        2        	|  4  	|   11,833   	|
|  CNN4  	|        13       	|        2        	|  4  	|   20,289   	|
|  CNN5  	|        13       	|        4        	|  4  	|   73,343   	|

## Compression techniques

We applied Knowledge Distillation, pruning, and quantization to our models to compress them. We used several values and methods to compress them. In each respective folder you can find a python notebook of how the compression techniques were carried out. Finally, in the models folder, you can find the models if you want to use them.

## References
<a id="1">[1]</a> 
Tan, Shawn, et al. "Icentia11k: An unsupervised representation learning dataset for arrhythmia subtype discovery." arXiv preprint arXiv:1910.09570 (2019).

<a id="2">[2]</a> 
https://github.com/shawntan/icentia-ecg

<a id="3">[3]</a> 
Towards atrial fibrillation detection in wearable devices using deep learning. (pending publishing)

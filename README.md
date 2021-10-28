# A Compressed Deep Learning Model to detect Cardiac Arrhythmias

ECG interpretation by deep learning models has proven to offer an important alternative in the clinical ECG workflow. 
However, these deep learning models are both computational and memory intensive, making them difficult to deploy on wearable devices because of their limitations on computational resources.  
A small model with appropriate accuracy will be more feasible to implement in a wearable device. 
We propose several model with state-of-the-art accuracy and F1-scores. The proposed models were evaluated on the Incentia11K dataset. Our CNN4 baseline model with only 11,833 parameters achieved an 0.9 F1-score which is higher than the average F1-score of a cardiologist (0.78).
The model predicts cardiac arrhythmias, normal rhythms, or noise signals from directly ECG signals without pre-processing.

## Dataset description

We chose the Icentia11K dataset [1], the largest public ECG dataset of continuous raw signals. It contains signals from 11,000 patients and has around 2 billion labeled beats. Our specific version has 2.5 million signals. Please refer to [1] and [2] for more information about downloading and using this dataset. We split the data into training, validation, and test sets with 64%, 16%, and 20% of the data, respectively.

## Baseline and Compact Models 

We created and trained several models variating the number of residual blocks, number of initial filters and number of outputs of each residual block, these models are better eplained in our paper []. In the training folder you can find a notebook where is explained in detail the process of creating and training the models. The next table shows how our five models were created.

| Models 	| Residual Blocks 	| Initial Filters 	| s_j 	| Parameters 	|
|:------:	|:---------------:	|:---------------:	|:---:	|:----------:	|
|  CNN1  	|        8        	|        2        	|  8  	|    4,455   	|
|  CNN2  	|        13       	|        2        	|  8  	|    7,093   	|
|  CNN3  	|        11       	|        2        	|  4  	|   11,833   	|
|  CNN4  	|        13       	|        2        	|  4  	|   20,289   	|
|  CNN5  	|        13       	|        4        	|  4  	|   73,343   	|

## Compression techniques

We applied Knowledge Distillation, pruning, and quantization to our models to compress them. We used several values and methods to compress them. In each respective folder you can find a notebook of how the compression techniques were carried out. Finally, in the models folder you can find the models if you want to use them. 

# A Compressed Deep Learning Model to detect Cardiac Arrhythmias

ECG interpretation by deep learning models has proven to offer an important alternative in the clinical ECG workflow. 
However, these deep learning models are both computational and memory intensive, making them difficult to deploy on wearable devices because of their limitations on computational resources.  
A small model with appropriate accuracy will be more feasible to implement in a wearable device. 
We propose several model with state-of-the-art accuracy and F1-scores. The proposed models were evaluated on the Incentia11K dataset. Our CNN4 baseline model with only 11,833 parameters achieved an 0.9 F1-score which is higher than the average F1-score of a cardiologist (0.78).
The model predicts cardiac arrhythmias, normal rhythms, or noise signals from directly ECG signals without pre-processing.

## Dataset description

We chose the Icentia11K dataset [1], the largest public ECG dataset of continuous raw signals. It contains signals from 11,000 patients and has around 2 billion labeled beats. Our specific version has 2.5 million signals. Please refer to [1] and [2] for more information about downloading and using this dataset. We split the data into training, validation, and test sets with 64%, 16%, and 20% of the data, respectively.

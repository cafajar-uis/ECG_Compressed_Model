# A Compressed Deep Learning Model to detect Cardiac Arrhythmias

ECG interpretation by deep learning models has proven to offer an important alternative in the clinical ECG workflow. 
However, these deep learning models are both computational and memory intensive, making them difficult to deploy on wearable devices because of their limitations on computational resources.  
A small model with appropriate accuracy will be more feasible to implement in a wearable device. 
We propose a model with around $50\times$ fewer parameters than accuracy-equivalent models. The proposed model was evaluated by 5-fold stratified cross-validation on the training data set provided by the PhysioNet/CinC Challenge 2017. We achieved an F1 score of $0.85 \pm 0.0064$.
The model predicts cardiac arrhythmias, normal rhythms, or noise signals from directly ECG signals without pre-processing.

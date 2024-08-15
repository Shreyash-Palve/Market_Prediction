# Market_Prediction
used a supervised autoencoder MLP approach and my teammates use XGBoost. Our final submission is a simple blend of these two models. Here, I would like to explain my approach in detail.

The supervised autoencoder approach was initially proposed in Bottleneck encoder + MLP + Keras Tuner 8601c5, where one supervised autoencoder is trained separately before cross-validation (CV) split. I have realised that this training may cause label leakage because the autoencoder has seen part of the data in the validation set in each CV split and it can generate label-leakage features to overfit. So, my approach is to train the supervised autoencoder along with MLP in one model in each CV split. The training processes and explanations are given in the notebook and the following statements.

Cross-Validation (CV) Strategy and Feature Engineering:

5-fold 31-gap purged group time-series split
Remove first 85 days for training since they have different feature variance
Forward-fill the missing values
Transfer all resp targets (resp, resp_1, resp_2, resp_3, resp_4) to action for multi-label classification
Use the mean of the absolute values of all resp targets as sample weights for training so that the model can focus on capturing samples with large absolute resp.
During inference, the mean of all predicted actions is taken as the final probability
Deep Learning Model:

Use autoencoder to create new features, concatenating with the original features as the input to the downstream MLP model
Train autoencoder and MLP together in each CV split to prevent data leakage
Add target information to autoencoder (supervised learning) to force it to generate more relevant features, and to create a shortcut for backpropagation of gradient
Add Gaussian noise layer before encoder for data augmentation and to prevent overfitting
Use swish activation function instead of ReLU to prevent ‘dead neuron’ and smooth the gradient
Batch Normalisation and Dropout are used for MLP
Train the model with 3 different random seeds and take the average to reduce prediction variance
Only use the models (with different seeds) trained in the last two CV splits since they have seen more data
Only monitor the BCE loss of MLP instead of the overall loss for early stopping
Use Hyperopt to find the optimal hyperparameter set

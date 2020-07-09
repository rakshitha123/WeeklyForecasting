# WeeklyForecasting

This repository contains the experiments related with a new baseline model that can be used in forecasting weekly time series. 
This model uses the forecasts of 4 sub-models: TBATS, Theta, Dynamic Harmonic Regression ARIMA and a global Recurrent Neural Network (RNN), and optimally combine them using lasso regression.


# Instructions for execution

## Obtain RNN Forecasts
First obtain the RNN forecasts for your weekly dataset. The RNN code is available in ./models/global_rnn.

For that you need to first preprocess your dataset separately for validation and testing phases. Create a folder named "datasets" in the parent level and place your datasets there.
The R scripts in "./models/global_rnn/preprocess_scripts" folder show an example of preprocessing for the NN5 weekly dataset.
Run the scripts named as "create_tfrecords.py" to convert the text data into a binary format. The generated tfrecords are used to train the RNN.

Add experiments into "./models/global_rnn/utility_scripts/execution_scripts/rnn_experiments.sh" script according to the format mentioned in the script and run experiments through that. Makesure to provide absolute paths for datasets.
First get the RNN forecasts corresponding with the test phase using "generic_model_trainer.py" and then get the RNN forecasts for the validation phase using "optimized_trainer.py".
See the examples provided in "rnn_experiments.sh" for more details.

The forecasts and errors will be stored in ./results/forecasts and ./results/errors folders respectively.

## Execute the Proposed Baseline Model
You can directly execute the exepriments, once you have obtained the RNN forecasts.
The exepriments are executed using the functions implemented in "weekly_experiments.R". It will calculate the forecasts of the remaining 3 sub-models and optimally combine them using lasso regression.
You can provide different options when executing models. E.g. training the lasso model with and without series features, transforming sub-model forecasts into log scale before training etc.
See the examples provided in "weekly_experiments.R" file for more details. 

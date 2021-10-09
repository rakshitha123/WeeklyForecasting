# WeeklyForecasting

This repository contains the experiments related with a new baseline model that can be used in forecasting weekly time series. 
This model uses the forecasts of 4 sub-models: TBATS, Theta, Dynamic Harmonic Regression ARIMA and a global Recurrent Neural Network (RNN), and optimally combine them using lasso regression.


# Instructions for execution

## Obtain RNN Forecasts
First obtain the RNN forecasts for your weekly dataset. The RNN code is available in ./models/global_rnn.

For that, you need to first preprocess your dataset separately for validation and testing phases. Create a folder named "datasets" in the parent level and place your datasets and the corresponding results files there.
The R scripts in "./models/global_rnn/preprocess_scripts" folder show examples of preprocessing for the NN5 weekly dataset (with fourier terms) and M4 weekly dataset (with seasonal lags).
After running the R scripts, run the scripts named as "create_tfrecords.py" with the parameters corresponding with your dataset to convert the text data into a binary format. The generated tfrecords are used to train the RNN.

For new datasets, add experiments into "./models/global_rnn/utility_scripts/execution_scripts/rnn_experiments.sh" script according to the format mentioned in the script and run experiments through that. Makesure to provide absolute paths for datasets.
First get the RNN forecasts corresponding with the test phase using "generic_model_trainer.py" and then get the RNN forecasts for the validation phase using "optimized_trainer.py".
See the examples provided in "rnn_experiments.sh" for more details.

Create a folder named "results" at the parent level and create 2 sub-folders within it named "forecasts" and "errors". The forecasts and errors provided by RNN will be stored into "./results/forecasts" and "./results/errors" folders respectively.

## Execute the Proposed Baseline Model
You can directly execute the exepriments related to the proposed baseline model, once you have obtained the RNN forecasts.
The exepriments are executed using the functions implemented in "weekly_experiments.R". It will calculate the forecasts of the remaining 3 sub-models, optimally combine all sub-model forecasts using lasso regression and calculate errors for the generated forecasts.
You can provide different options when executing models. E.g. using different meta-learning algorithms to combine forecasts, using different base model pools, training the meta-learning models with and without series features, transforming sub-model forecasts into log scale before training etc.
The forecasts and errors provided by our model will be stored into "./results/forecasts" and "./results/errors" folders respectively.
See the examples provided in "weekly_experiments.R" file for more details.

Now you can also execute our proposed forecasting model using the R script, "./weekly_forecasting_model.R". First obtain the RNN forecasts as explained above. Then, you only have to change a set of parameters in the "./weekly_forecasting_model.R" script such as the name and path of your dataset and forecast horizon. The script will automatically provides the forecasts of our proposed weekly forecasting model.

# Experimental Datasets
The experimental weekly datasets are available in the Google Drive folder at this [link](https://drive.google.com/drive/folders/109-ZYZAHQU1YLQfVLDnpgT4MRX_CqINH?usp=sharing)


# Citing Our Work
When using this repository, please cite:

```{r} 
@misc{godahewa2020weekly,
  title = {A Strong Baseline for Weekly Time Series Forecasting},
  author = {Godahewa, Rakshitha and Bergmeir, Christoph and Webb, Geoffrey I. and Montero-Manso, Pablo},
  howPublished = {https://arxiv.org/abs/2010.08158},
  year = {2020}
}
```



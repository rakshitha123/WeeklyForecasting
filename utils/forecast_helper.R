# Helper functions that are required for weekly experiments

# Function to create training and test sets
# Parameters
# training_path - The file path of the training dataset
# test_path - The file path of the actual results corresponding with the expected forecasts
create_traing_test_sets <- function(training_path, test_path){
  training_set <- readLines(training_path)
  training_set <- strsplit(training_set, ',')
  
  test_set <- read.csv(test_path, header = FALSE, sep = ";")
  test_set <- test_set[-1]
  
  list(training_set, test_set)
}


# Function to convert a dataframe into a vector in column-wise
# Parameters
# df - A dataframe that needs to be converted into a vector
create_vector <- function(df){
  vec <- c()
  for(i in 1:ncol(df)){
    vec <- c(vec, as.numeric(df[,i]))
  }
  vec
}


# Function to create a list containing the actual results corresponding with the validation phase and forecasts of all sub-models corresponding with validation and test phases
# Parameters
# training_set - A dataframe containing the training series
# forecast_horizon - Expected forecast horizon
# phases - Phase names: validation and test
# methods - Sub-model names (dhr_arima, tbats, theta, rnn)
# file_prefix - Prefix of forecast files
load_forecasts <- function(training_set, forecast_horizon, phases, methods, file_prefix){
  training_forecasts <- matrix(NA, nrow = length(training_set), ncol = forecast_horizon)
  result <- list()
  
  for(i in 1:length(training_set)){
    time_series <- as.numeric(unlist(training_set[i], use.names = FALSE))
    training_forecasts[i,] <- tail(time_series, forecast_horizon)
  }
  
  result[["training_forecasts"]] <- training_forecasts
  
  # Load pre-calculated forecasts
  for(method in methods){
    for(phase in phases)
      result[[paste0(method, "_", phase, "_forecasts")]] <- read.csv(paste0(file_prefix, "_", method, "_", phase, "_forecasts.txt"), header = FALSE, sep = " ")
  }
  
  result
}

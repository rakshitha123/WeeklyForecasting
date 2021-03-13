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


# Function to calculate sub-model forecasts for a given dataset
# This returns a list containing the actual results corresponding with the validation phase and forecasts of all sub-models corresponding with validation and test phases
# Parameters
# training_set - A dataframe containing the training series
# forecast_horizon - The expected forecast horizon
# dataset_name - A string value for the dataset name. This will be used to as the prefix of the forecasts files
# optimal_k_value - The k value that should be used when running dynamic harmonic regression arima (dhr_arima) model
# integer_conversion - Whether the forecasts should be rounded or not
# write_sub_model_forecasts - Whether the sub-model forecasts should be separately written into files
do_forecasting <- function(training_set, forecast_horizon, dataset_name, optimal_k_value = 1, integer_conversion = FALSE, write_sub_model_forecasts = FALSE){
  
  training_forecasts <- matrix(NA, nrow = length(training_set), ncol = forecast_horizon)
  
  for(method in METHODS){
    for(phase in PHASES){
      assign(paste0(method, "_", phase, "_forecasts"), NULL)
    }
  }
  
  for(i in 1:length(training_set)){
    print(i)
    
    time_series <- as.numeric(unlist(training_set[i], use.names = FALSE))
    
    training_forecasts[i,] <- tail(time_series, forecast_horizon)
    
    # The last set of values (equal to the forecast horizon) of each series are taken for validation
    validation_series <- ts(time_series[1:(length(time_series) - forecast_horizon)], frequency = SEASONALITY)
    test_series <- ts(time_series, frequency = SEASONALITY)
    
    for(method in METHODS){
      for(phase in PHASES){
        if(method %in% GLOBAL_MODELs){
          # The rnn forecasts should have obtained beforehand and the corresponding forecast files should be inside FORECASTS_DIR with the name <dataset_name>_rnn_<phase>_forecasts.txt 
          forecast_path <- file.path(FORECASTS_DIR, paste0(dataset_name, "_", method, "_", phase, "_forecasts.txt"))
          assign(paste0(method, "_", phase, "_forecasts"), rbind(eval(parse(text = paste0(method, "_", phase, "_forecasts"))), eval(parse(text = paste0("get_", method, "_forecasts", "(forecast_path, i)")))))
        }
        else if(method == "dhr_arima")
          assign(paste0(method, "_", phase, "_forecasts"), rbind(eval(parse(text = paste0(method, "_", phase, "_forecasts"))), eval(parse(text = paste0("get_", method, "_forecasts", "(", phase, "_series, forecast_horizon, optimal_k_value)")))))
        else
          assign(paste0(method, "_", phase, "_forecasts"), rbind(eval(parse(text = paste0(method, "_", phase, "_forecasts"))), eval(parse(text = paste0("get_", method, "_forecasts", "(", phase, "_series, forecast_horizon)")))))
      }
    }
  }
  
  result <- list()
  result[["training_forecasts"]] <- training_forecasts
  
  for(method in METHODS){
    for(phase in PHASES){
      forecasts <- eval(parse(text = paste0(method, "_", phase, "_forecasts")))
      
      forecasts[forecasts < 0] <- 0
      
      # Round forecasts if required
      if(integer_conversion)
        forecasts <- round(forecasts)
      
      result[[paste0(method, "_", phase, "_forecasts")]] <- forecasts
      
      # Write sub-model forecasts if required
      if(write_sub_model_forecasts)
        write.table(forecasts, file.path(FORECASTS_DIR, paste0(dataset_name, "_", method, "_", phase, "_forecasts.txt"), fsep = "/"), row.names = FALSE, col.names = FALSE, quote = FALSE)
    }
  }
  
  result
}


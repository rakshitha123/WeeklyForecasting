# Functions required for feature calculation

# Parameters
# training_set - A dataframe containing the training series
# forecast_horizon - Expected forecast horizon
# seasonality - Frequency of the dataset (365.25/7)
# num_of_features - Number of features (42)
calculate_features <- function(training_set, forecast_horizon, seasonality, num_of_features){
  tryCatch({
    ts_list <- list()
    
    for(i in 1:length(training_set)){
      series <- ts(as.numeric(unlist(training_set[i], use.names = FALSE)), freq = seasonality)
      ts_list[[i]] <- list(st = paste0("D", i), x = series, n = length(series), h = forecast_horizon)
    }
    
    validation_ts_list <- M4metalearning:::temp_holdout(ts_list)
    
    validation_features <- M4metalearning:::THA_features(validation_ts_list, n.cores = 3)
    test_features <- M4metalearning:::THA_features(ts_list, n.cores = 3)
    
    validation_feature_info <- fill_features(validation_features, num_of_features)
    test_feature_info <- fill_features(test_features, num_of_features)
    
    list(validation_feature_info, test_feature_info)
  }, error = function(e){
    print(e)
    return(e)
  })
}


# Function to create a matrix with the calculated features
# Parameters
# feature_data - The output containing the features provided by THA_features function in M4metalearning package
# num_of_features - Number of features (42)
fill_features <- function(feature_data, num_of_features){
  features_info <- matrix(nrow = length(feature_data), ncol = num_of_features)
  
  for(i in 1:length(feature_data)){
    tryCatch(
      features_info[i,] <- c(as.numeric(feature_data[[i]]$features))
      , error = function(e) {
        print(e)
        features_info[i,] <- rep(0, num_of_features)
      }
    )
  }
  
  features_info
}

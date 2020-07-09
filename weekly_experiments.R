set.seed(123)


# Define paths and other constant variables
BASE_DIR <- "WeeklyForecasting"
FORECASTS_DIR <- file.path(BASE_DIR, "results", "forecasts", fsep = "/")
ERRORS_DIR <- file.path(BASE_DIR, "results", "errors", fsep = "/")
SEASONALITY <- 365.25/7
GLOBAL_MODELs <- c("rnn")
LOCAL_MODELS <- c("dhr_arima", "tbats", "theta")
METHODS <- c(LOCAL_MODELS, GLOBAL_MODELs)
LAMBDAS_TO_TRY <- 10^seq(-3, 5, length.out = 100)
PHASES <- c("validation", "test")


# Use for experiments with features
NUM_OF_FEATURES <- 42


# Execute helper scripts
source(file.path(BASE_DIR, "models", "sub_model_forecast_calculator.R", fsep = "/"))
source(file.path(BASE_DIR, "utils", "error_calculator.R", fsep = "/"))
source(file.path(BASE_DIR, "utils", "feature_calculator.R", fsep = "/"))
source(file.path(BASE_DIR, "utils", "forecast_helper.R", fsep = "/"))


# Function to calculate sub-model forecasts for a given dataset
# This returns a list containing the actual results corresponding with the validation phase and forecasts of all sub-models corresponding with validation and test phases
# Parameters
# training_set - A dataframe containing the training series
# forecast_horizon - The expected forecast horizon
# dataset_name - A string value for the dataset name. This will be used to as the prefix of the forecasts files
# optimal_k_value - The k value that should be used when running dynamic harmonic regression arima (dhr_arima) model
# integer_conversion - Whether the forecasts should be rounded or not
# write_sub_model_forecasts - Whether the sub-model forecasts should be separately written into files
do_forecasting <- function(training_set, forecast_horizon, dataset_name, optimal_k_value, integer_conversion = FALSE, write_sub_model_forecasts = FALSE){
  
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


# Function to optimally combine a set of sub-model forecasts using lasso regression. It will train a single model to obtain final forecasts.
# Parameters
# training_set_path - The file path of the training dataset
# test_set_path - The file path of the actual results corresponding with the expected forecasts. This will be used for final error calculation
# forecast_horizon - The expected forecast horizon
# dataset_name - A string value for the dataset name. This will be used to as the prefix of the forecasts/error files
# optimal_k_value - The k value that should be used when running dynamic harmonic regression arima (dhr_arima) model
# log_transformation - Whether the sub-model forecasts should be transformed into log scale before training the lasso regression model
# use_features - Whether the series features should be used to train the lasso regression model
# calculate_sub_model_forecasts - Whether the sub-models forecasts should be calculated or not. If you have already calculated the sub-model forecasts set this as FALSE and place the forecast files inside the FORECASTS_DIR with the name <dataset_name>_<method_name>_<phase>_forecasts.txt 
# write_sub_model_forecasts - Whether the sub-model forecasts should be separately written into files. This will be useful when calculating the sub-model forecasts for the first time
# integer_conversion - Whether the forecasts should be rounded or not
# address_near_zero_instability - Whether the results can have zeros or not. This will be used when calculating smape
# forecasts_to_be_replaced - If you want to replace some of the final forecasts with some set of new values, they should be provided here
# replacable_row_indexes - The series indexes corresponding with the replacable forecasts provided using forecasts_to_be_replaced 
# trainable_row_indexes - If you need to use consider only a set of series when training the lasso model, provide the required training series indexes with this parameter.
#                         Useful with m4 weekly dataset as the feature calculation is only possible for a set of series. Only those series will be used when training the lasso model
run_single_model_lasso_regression <- function(training_set_path, test_set_path, forecast_horizon, dataset_name, optimal_k_value, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, integer_conversion = FALSE, address_near_zero_instability = FALSE, forecasts_to_be_replaced = NULL, replacable_row_indexes = NULL, trainable_row_indexes = NULL){
  
  output_file_name <- dataset_name
  
  print("Loading data")
  
  # Obtain traning and test sets
  training_test_sets <- create_traing_test_sets(training_set_path, test_set_path)
  training_set <- training_test_sets[[1]]
  test_set <- training_test_sets[[2]]
  
  df_forecast <- matrix(nrow = length(training_set), ncol = forecast_horizon)
  
  # If trainable_row_indexes is not provided, then use all series for training
  if(is.null(trainable_row_indexes))
    trainable_row_indexes <- 1:length(training_set)
  
  full_train_df <- NULL
  full_test_df <- NULL
  
  # Calculate features. The same features are used as in FFORMA
  if(use_features){
    output_file_name <- paste0(output_file_name, "_with_features")
    
    print("Started calculating features")
    
    feature_info <- calculate_features(training_set, forecast_horizon, SEASONALITY, NUM_OF_FEATURES)
    feature_train_df <- feature_info[[1]]
    feature_test_df <- feature_info[[2]]
    
    #Arrange features to build a single lasso regression model
    for(i in 1:forecast_horizon){
      full_train_df <- rbind(full_train_df, feature_train_df[trainable_row_indexes,])
      full_test_df <- rbind(full_test_df, feature_test_df[trainable_row_indexes,])
    }
    
    print("Finished calculating features")
  }
  
  # Calculate sub-model forecasts if required. Otherwise, load the pre-calculated forecasts
  if(calculate_sub_model_forecasts){
    print("Started calculating sub-model forecasts")
    all_forecast_matrices <- do_forecasting(training_set, forecast_horizon, dataset_name, optimal_k_value, integer_conversion, write_sub_model_forecasts)
    print("Finished calculating sub-model forecasts")
  }else{
    all_forecast_matrices <- load_forecasts(training_set, forecast_horizon, PHASES, METHODS, file.path(FORECASTS_DIR, dataset_name, fsep = "/"))
  }
  
  for(method in METHODS){
    validation_forecasts <- eval(parse(text = paste0("all_forecast_matrices[['", method, "_validation_forecasts']]")))
    test_forecasts <- eval(parse(text = paste0("all_forecast_matrices[['", method, "_test_forecasts']]")))
    
    # Transform forecasts into log scale if required
    if(log_transformation){
      validation_forecasts <- log(validation_forecasts + 1)
      test_forecasts <- log(test_forecasts + 1)
    }
    
    # Aggregate forecasts with the training and test dataframes after converting them into vectors
    full_train_df <- cbind(full_train_df, create_vector(validation_forecasts[trainable_row_indexes,]))
    full_test_df <- cbind(full_test_df, create_vector(test_forecasts[trainable_row_indexes,]))
  }
  
  # Modify column names of train and test dataframes
  if(use_features){
    colnames(full_train_df) <- sprintf("X%s", seq(1:(NUM_OF_FEATURES + length(METHODS))))
    colnames(full_test_df) <- sprintf("X%s", seq(1:(NUM_OF_FEATURES + length(METHODS))))
  }else{
    colnames(full_train_df) <- sprintf("X%s",seq(1:length(METHODS)))
    colnames(full_test_df) <- sprintf("X%s",seq(1:length(METHODS)))
  }
  
  full_train_df <- as.matrix(full_train_df)
  full_test_df <- as.matrix(full_test_df)
  
  train_forecasts <- all_forecast_matrices[["training_forecasts"]]
  train_y <- create_vector(train_forecasts[trainable_row_indexes,])
  
  if(log_transformation){
    output_file_name <- paste0(output_file_name, "_log")
    train_y <- log(train_y + 1)
  }
  
  train_y <- as.matrix(train_y)
  
  # Find the best lambda value to be used with 10-fold cross validation
  print("Processing ensemble model")
  
  lasso_cv <- glmnet:::cv.glmnet(full_train_df, train_y, alpha = 1, lambda = LAMBDAS_TO_TRY, standardize = TRUE, nfolds = 10)   
  lambda_cv <- lasso_cv$lambda.min
  print(paste0("Optimised lambda = ", lambda_cv))
  
  # Train lasso regression model
  final_model <- glmnet:::glmnet(full_train_df, train_y, alpha = 1, lambda = lambda_cv, standardize = TRUE)
  
  # Obtain predictions
  predictions <- predict(final_model, full_test_df)
  
  # Rescale predictions if required
  if(log_transformation)
    converted_predictions <- exp(predictions) - 1
  else
    converted_predictions <- predictions
  
  # Rearrange forecasts into a matrix
  df_forecast[trainable_row_indexes,] <- matrix(converted_predictions, ncol = forecast_horizon, byrow = FALSE)
  
  # Replace forecasts if required
  if(!is.null(forecasts_to_be_replaced) & !is.null(replacable_row_indexes))
    df_forecast[replacable_row_indexes,] <- data.matrix(forecasts_to_be_replaced, rownames.force = NA)
  
  df_forecast[df_forecast < 0] <- 0
  
  # Round final forecasts if required
  if(integer_conversion)
    df_forecast <- round(df_forecast)
  
  write.table(df_forecast, file.path(FORECASTS_DIR, paste0(output_file_name, "_single_model_lasso_forecasts.txt"), fsep = "/"), row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  # Calculate errors
  print("Calculating errors")
  calculate_errors(df_forecast, test_set, training_set, SEASONALITY, file.path(ERRORS_DIR, paste0(output_file_name, "_single_model"), fsep = "/"), address_near_zero_instability)
}


# Function to optimally combine a set of sub-model forecasts using lasso regression. It will train a separate model per each horizon to obtain final forecasts.
# Parameters
# training_set_path - The file path of the training dataset
# test_set_path - The file path of the actual results corresponding with the expected forecasts. This will be used for final error calculation
# forecast_horizon - The expected forecast horizon
# dataset_name - A string value for the dataset name. This will be used to as the prefix of the forecasts/error files
# optimal_k_value - The k value that should be used when running dynamic harmonic regression arima (dhr_arima) model
# log_transformation - Whether the sub-model forecasts should be transformed into log scale before training the lasso regression model
# use_features - Whether the series features should be used to train the lasso regression model
# calculate_sub_model_forecasts - Whether the sub-models forecasts should be calculated or not. If you have already calculated the sub-model forecasts set this as FALSE and place the forecast files inside the FORECASTS_DIR with the name <dataset_name>_<method_name>_<phase>_forecasts.txt 
# write_sub_model_forecasts - Whether the sub-model forecasts should be separately written into files. This will be useful when calculating the sub-model forecasts for the first time
# integer_conversion - Whether the forecasts should be rounded or not
# address_near_zero_instability - Whether the results can have zeros or not. This will be used when calculating SMAPE
# forecasts_to_be_replaced - If you want to replace some of the final forecasts with some set of new values, they should be provided here
# replacable_row_indexes - The series indexes corresponding with the replacable forecasts provided using forecasts_to_be_replaced 
# trainable_row_indexes - If you need to consider only a set of series when training the lasso model, provide the required training series indexes with this parameter
#                         Useful with m4 weekly dataset as the feature calculation is only possible for a set of series. Only those series will be used when training the lasso model
run_per_horizon_lasso_regression <- function(training_set_path, test_set_path, forecast_horizon, dataset_name, optimal_k_value, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, integer_conversion = FALSE, address_near_zero_instability = FALSE, forecasts_to_be_replaced = NULL, replacable_row_indexes = NULL, trainable_row_indexes = NULL){
  
  output_file_name <- dataset_name
  
  print("Loading data")
  
  # Obtain traning and test sets
  training_test_sets <- create_traing_test_sets(training_set_path, test_set_path)
  training_set <- training_test_sets[[1]]
  test_set <- training_test_sets[[2]]
  
  df_forecast <- matrix(nrow = length(training_set), ncol = forecast_horizon)
  
  # If trainable_row_indexes is not provided, then use all series for training
  if(is.null(trainable_row_indexes))
    trainable_row_indexes <- 1:length(training_set)
  
  # Calculate features. The same features are used as in FFORMA
  if(use_features){
    output_file_name <- paste0(output_file_name, "_with_features")
    
    print("Started calculating features")
    
    feature_info <- calculate_features(training_set, forecast_horizon, SEASONALITY, NUM_OF_FEATURES)
    feature_train_df <- feature_info[[1]]
    feature_test_df <- feature_info[[2]]
    
    print("Finished calculating features")
  }
  
  # Calculate sub-model forecasts if required. Otherwise, load the pre-calculated forecasts
  if(calculate_sub_model_forecasts){
    print("Started calculating sub-model forecasts")
    all_forecast_matrices <- do_forecasting(training_set, forecast_horizon, dataset_name, optimal_k_value, integer_conversion, write_sub_model_forecasts)
    print("Finished calculating sub-model forecasts")
  }else{
    all_forecast_matrices <- load_forecasts(training_set, forecast_horizon, PHASES, METHODS, file.path(FORECASTS_DIR, dataset_name, fsep = "/"))
  }
  
  # For each horizon train a separate lasso regression model
  for(f in 1:forecast_horizon){
    full_train_df <- NULL
    full_test_df <- NULL
    
    if(use_features){
      full_train_df <- feature_train_df[trainable_row_indexes,]
      full_test_df <- feature_test_df[trainable_row_indexes,]
    }
    
    for(method in METHODS){
      validation_forecasts <- eval(parse(text = paste0("all_forecast_matrices[['", method, "_validation_forecasts']]")))
      test_forecasts <- eval(parse(text = paste0("all_forecast_matrices[['", method, "_test_forecasts']]")))
      
      # Transform forecasts into log scale if required
      if(log_transformation){
        validation_forecasts[,f] <- log(validation_forecasts[,f] + 1)
        test_forecasts[,f] <- log(test_forecasts[,f] + 1)
      }
      
      full_train_df <- cbind(full_train_df, validation_forecasts[trainable_row_indexes, f])
      full_test_df <- cbind(full_test_df, test_forecasts[trainable_row_indexes, f])
    }
    
    # Modify column names of train and test dataframes
    if(use_features){
      colnames(full_train_df) <- sprintf("X%s", seq(1:(NUM_OF_FEATURES + length(METHODS))))
      colnames(full_test_df) <- sprintf("X%s", seq(1:(NUM_OF_FEATURES + length(METHODS))))
    }else{
      colnames(full_train_df) <- sprintf("X%s",seq(1:length(METHODS)))
      colnames(full_test_df) <- sprintf("X%s",seq(1:length(METHODS)))
    }
    
    full_train_df <- as.matrix(full_train_df)
    full_test_df <- as.matrix(full_test_df)
    
    train_forecasts <- all_forecast_matrices[["training_forecasts"]]
    train_y <- train_forecasts[trainable_row_indexes, f]
    
    if(log_transformation)
      train_y <- log(train_y + 1)
    
    train_y <- as.matrix(train_y)
    
    # Find the best lambda value to be used with 10-fold cross validation
    print(paste0("Processing ensemble model for horizon ", f))
    
    lasso_cv <- glmnet:::cv.glmnet(full_train_df, train_y, alpha = 1, lambda = LAMBDAS_TO_TRY, standardize = TRUE, nfolds = 10)   
    lambda_cv <- lasso_cv$lambda.min
    print(paste0("Optimised lambda for horizon ", f, " = ", lambda_cv))
    
    # Train lasso regression model
    final_model <- glmnet:::glmnet(full_train_df, train_y, alpha = 1, lambda = lambda_cv, standardize = TRUE)
    
    # Obtain predictions
    predictions <- predict(final_model, full_test_df)
    
    # Rescale predictions if required
    if(log_transformation)
      converted_predictions <- exp(predictions) - 1
    else
      converted_predictions <- predictions
    
    df_forecast[trainable_row_indexes, f] <- converted_predictions
  }
  
  # Replace forecasts if required
  if(!is.null(forecasts_to_be_replaced) & !is.null(replacable_row_indexes)){
    df_forecast[replacable_row_indexes,] <- data.matrix(forecasts_to_be_replaced, rownames.force = NA)
  }
  
  df_forecast[df_forecast < 0] <- 0
  
  # Round final forecasts if required
  if(integer_conversion)
    df_forecast <- round(df_forecast)
  
  if(log_transformation)
    output_file_name <- paste0(output_file_name, "_log")
  
  write.table(df_forecast, file.path(FORECASTS_DIR, paste0(output_file_name, "_per_horizon_lasso_forecasts.txt"), fsep = "/"), row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  # Calculate errors
  print("Calculating errors")
  calculate_errors(df_forecast, test_set, training_set, SEASONALITY, file.path(ERRORS_DIR, paste0(output_file_name, "_per_horizon"), fsep = "/"), address_near_zero_instability)
}


# Sample Usage

# NN5 Weekly
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE)
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)

run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)


# Ausgrid Weekly
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE)
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)

run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)


#Kaggle Web Traffic Weekly
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)

run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)


#M4 Weekly
snaive_forecasts <- read.csv(file.path(FORECASTS_DIR, "m4_weekly_snaive.txt"), header = FALSE)

run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "m4_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "m4_weekly_results.txt"), 13, "m4_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, forecasts_to_be_replaced = snaive_forecasts[295:359,], replacable_row_indexes = 295:359, trainable_row_indexes = 1:294)
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "m4_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "m4_weekly_results.txt"), 13, "m4_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, forecasts_to_be_replaced = snaive_forecasts[295:359,], replacable_row_indexes = 295:359, trainable_row_indexes = 1:294)
run_single_model_lasso_regression(file.path(BASE_DIR, "datasets", "m4_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "m4_weekly_results.txt"), 13, "m4_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, forecasts_to_be_replaced = snaive_forecasts[295:359,], replacable_row_indexes = 295:359, trainable_row_indexes = 1:294)

run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "m4_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "m4_weekly_results.txt"), 13, "m4_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, forecasts_to_be_replaced = snaive_forecasts[295:359,], replacable_row_indexes = 295:359, trainable_row_indexes = 1:294)
run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "m4_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "m4_weekly_results.txt"), 13, "m4_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, forecasts_to_be_replaced = snaive_forecasts[295:359,], replacable_row_indexes = 295:359, trainable_row_indexes = 1:294)
run_per_horizon_lasso_regression(file.path(BASE_DIR, "datasets", "m4_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "m4_weekly_results.txt"), 13, "m4_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, forecasts_to_be_replaced = snaive_forecasts[295:359,], replacable_row_indexes = 295:359, trainable_row_indexes = 1:294)



set.seed(123)


# Define paths and other constant variables
BASE_DIR <- "WeeklyForecasting"
FORECASTS_DIR <- file.path(BASE_DIR, "results", "forecasts", fsep = "/")
ERRORS_DIR <- file.path(BASE_DIR, "results", "errors", fsep = "/")
SEASONALITY <- 365.25/7 # Change this to 7 for daily datasets
GLOBAL_MODELs <- c("rnn")
LOCAL_MODELS <- c("dhr_arima", "tbats", "theta")
FULL_EXTRA_MODELS <- c("ets", "naive", "snaive", "rw_drift", "stlmar")
ORIGINAL_MODELS <- c("ets", "naive", "snaive", "rw_drift", "stlmar", "tbats", "theta", "arima", "nnetar") 
METHODS <- c(LOCAL_MODELS, GLOBAL_MODELs) # Change this accordingly to run experiments with different model pools
LAMBDAS_TO_TRY <- 10^seq(-3, 5, length.out = 100)
PHASES <- c("validation", "test")


# Use for experiments with features
NUM_OF_FEATURES <- 42


# Execute helper scripts
source(file.path(BASE_DIR, "models", "sub_model_forecast_calculator.R", fsep = "/"))
source(file.path(BASE_DIR, "utils", "error_calculator.R", fsep = "/"))
source(file.path(BASE_DIR, "utils", "feature_calculator.R", fsep = "/"))
source(file.path(BASE_DIR, "utils", "forecast_helper.R", fsep = "/"))


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
# feature_freqency - Frequency that should be used when calculating features. This should be set to 1 when calculating features for short series whose lengths are less than 2 seasonal cycles
# integer_conversion - Whether the forecasts should be rounded or not
# address_near_zero_instability - Whether the results can have zeros or not. This will be used when calculating sMAPE
# forecasts_to_be_replaced - If you want to replace some of the final forecasts with some set of new values, you can provide them here
# replacable_row_indexes - The series indexes corresponding with the replacable forecasts provided using forecasts_to_be_replaced 
# trainable_row_indexes - If you need to consider only a set of series when training the lasso model, then provide the required training series indexes with this parameter.
# seasonality - Seasonality that should be used to calculate MASE
# regression_type - type of regression used with ensembling, lasso, linear or xgboost
run_single_model_regression <- function(training_set_path, test_set_path, forecast_horizon, dataset_name, optimal_k_value, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, feature_freqency = SEASONALITY, integer_conversion = FALSE, address_near_zero_instability = FALSE, forecasts_to_be_replaced = NULL, replacable_row_indexes = NULL, trainable_row_indexes = NULL, seasonality = SEASONALITY, regression_type = "lasso"){
  
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
  
  start_time <- Sys.time()
  
  # Calculate features. The same features are used as in FFORMA
  if(use_features){
    output_file_name <- paste0(output_file_name, "_with_features")
    
    print("Started calculating features")
    
    feature_info <- calculate_features(training_set, forecast_horizon, feature_freqency, NUM_OF_FEATURES)
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
  colnames(train_y) <- "y"
  
  print("Processing ensemble model")
  
  if(regression_type == "lasso"){
    # Find the best lambda value to be used with 10-fold cross validation
    lasso_cv <- glmnet:::cv.glmnet(full_train_df, train_y, alpha = 1, lambda = LAMBDAS_TO_TRY, standardize = TRUE, nfolds = 10, lower.limits = rep(0, ncol(full_train_df)), upper.limits = rep(1, ncol(full_train_df)))   
    lambda_cv <- lasso_cv$lambda.min
    print(paste0("Optimised lambda = ", lambda_cv))
    
    # Train a lasso regression model
    final_model <- glmnet:::glmnet(full_train_df, train_y, alpha = 1, lambda = lambda_cv, standardize = TRUE, lower.limits = rep(0, ncol(full_train_df)), upper.limits = rep(1, ncol(full_train_df)))
    print(coef(final_model))
  }else if(regression_type == "linear"){
    fitting_data <- as.data.frame(cbind(train_y, full_train_df))
      
    formula <- "y ~ "
    for(predictor in 2:ncol(fitting_data)){
      if(predictor != ncol(fitting_data))
        formula <- paste0(formula, colnames(fitting_data)[predictor], " + ")
      else
        formula <- paste0(formula, colnames(fitting_data)[predictor])
    }
    
    formula <- paste(formula, "+ 0", sep = "")
    formula <- as.formula(formula)
    
    # Train a linear regression model
    final_model <- glm(formula = formula, data = fitting_data)
  }else if(regression_type == "xgboost"){
    # Define parameters grid to tune hyperparameters
    parameters_df = expand.grid(max_depth = seq(3, 10, 1), eta = seq(0.1, 0.5, 0.1))
    lowest_error_list = list()
    
    # Hyperparameter tuning using grid search
    for (row in 1:nrow(parameters_df)){
      xgcv <- xgboost:::xgboost(data = full_train_df,
                                label = train_y,
                                booster = "gbtree",
                                objective = "reg:linear",
                                eval_metric = "rmse",
                                max_depth = parameters_df$max_depth[row],
                                eta = parameters_df$eta[row],
                                nrounds = 250,
                                early_stopping_rounds = 30,
                                print_every_n = 10)
      
      lowest_error <- data.frame("min_error" = min(xgcv$evaluation_log$train_rmse))
      lowest_error_list[[row]] <- lowest_error
    }
    
    lowest_error_df <- do.call(rbind, lowest_error_list)
    grid_search_output <- cbind(lowest_error_df, parameters_df)
    optimised_parameters <- grid_search_output[grid_search_output$min_error == min(grid_search_output$min_error),]
    
    print(optimised_parameters)
    
    # Train a XGBoost model
    final_model <- xgboost:::xgboost(data = full_train_df,
                                     label = train_y,
                                     booster = "gbtree",
                                     objective = "reg:linear",
                                     eval_metric = "rmse",
                                     max_depth = optimised_parameters$max_depth,
                                     eta = optimised_parameters$eta,
                                     nround = 250,
                                     early_stopping_rounds = 30)
  }
 
  # Obtain predictions
  if(regression_type == "lasso")
    predictions <- predict(final_model, full_test_df)
  else if(regression_type == "linear")
    predictions <- predict.glm(final_model, as.data.frame(full_test_df))
  else if(regression_type == "xgboost")
    predictions <- predict(final_model, data.matrix(full_test_df))
  
  # Rescale predictions if required
  if(log_transformation)
    converted_predictions <- exp(predictions) - 1
  else
    converted_predictions <- predictions
  
  # Rearrange forecasts into a matrix
  df_forecast[trainable_row_indexes,] <- matrix(converted_predictions, ncol = forecast_horizon, byrow = FALSE)
  
  # Replace forecasts if you want
  if(!is.null(forecasts_to_be_replaced) & !is.null(replacable_row_indexes))
    df_forecast[replacable_row_indexes,] <- data.matrix(forecasts_to_be_replaced, rownames.force = NA)
  
  df_forecast[df_forecast < 0] <- 0
  
  # Round final forecasts if required
  if(integer_conversion)
    df_forecast <- round(df_forecast)
  
  write.table(df_forecast, file.path(FORECASTS_DIR, paste0(output_file_name, "_", regression_type, "_single_model_forecasts.txt"), fsep = "/"), row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  end_time <- Sys.time()
  
  # Execution time
  exec_time <- end_time - start_time
  print(paste0("Execution time: ", exec_time, " ", attr(exec_time, "units")))
  
  # Calculate errors
  print("Calculating errors")
  calculate_errors(df_forecast, test_set, training_set, seasonality, file.path(ERRORS_DIR, paste0(output_file_name, "_", regression_type, "_single_model"), fsep = "/"), address_near_zero_instability)
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
# feature_freqency - Frequency that should be used when calculating features. This should be set to 1 when calculating features for short series whose lengths are less than 2 seasonal cycles
# integer_conversion - Whether the forecasts should be rounded or not
# address_near_zero_instability - Whether the results can have zeros or not. This will be used when calculating sMAPE
# forecasts_to_be_replaced - If you want to replace some of the final forecasts with some set of new values, you can provide them here
# replacable_row_indexes - The series indexes corresponding with the replacable forecasts provided using forecasts_to_be_replaced 
# trainable_row_indexes - If you need to consider only a set of series when training the lasso model, then provide the required training series indexes with this parameter
# seasonality - Seasonality that should be used to calculate MASE
# regression_type - type of regression used with ensembling, lasso, linear or xgboost
run_per_horizon_regression <- function(training_set_path, test_set_path, forecast_horizon, dataset_name, optimal_k_value, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, feature_freqency = SEASONALITY, integer_conversion = FALSE, address_near_zero_instability = FALSE, forecasts_to_be_replaced = NULL, replacable_row_indexes = NULL, trainable_row_indexes = NULL, seasonality = SEASONALITY, regression_type = "lasso"){
  
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
    
    feature_info <- calculate_features(training_set, forecast_horizon, feature_freqency, NUM_OF_FEATURES)
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
    colnames(train_y) <- "y"
    
    print(paste0("Processing ensemble model for horizon ", f))
    
    if(regression_type == "lasso"){
      # Find the best lambda value to be used with 10-fold cross validation
      lasso_cv <- glmnet:::cv.glmnet(full_train_df, train_y, alpha = 1, lambda = LAMBDAS_TO_TRY, standardize = TRUE, nfolds = 10, lower.limits = rep(0, ncol(full_train_df)), upper.limits = rep(1, ncol(full_train_df)))   
      lambda_cv <- lasso_cv$lambda.min
      print(paste0("Optimised lambda = ", lambda_cv))
      
      # Train a lasso regression model
      final_model <- glmnet:::glmnet(full_train_df, train_y, alpha = 1, lambda = lambda_cv, standardize = TRUE, lower.limits = rep(0, ncol(full_train_df)), upper.limits = rep(1, ncol(full_train_df)))
    }else if(regression_type == "linear"){
      fitting_data <- as.data.frame(cbind(train_y, full_train_df))
      
      formula <- "y ~ "
      for(predictor in 2:ncol(fitting_data)){
        if(predictor != ncol(fitting_data))
          formula <- paste0(formula, colnames(fitting_data)[predictor], " + ")
        else
          formula <- paste0(formula, colnames(fitting_data)[predictor])
      }
      
      formula <- paste(formula, "+ 0", sep = "")
      formula <- as.formula(formula)
      
      # Train a linear regression model
      final_model <- glm(formula = formula, data = fitting_data)
    }else if(regression_type == "xgboost"){
      # Define parameters grid to tune hyperparameters
      # parameters_df = expand.grid(max_depth = seq(3, 10, 1), eta = seq(0.1, 0.5, 0.1), subsample = seq(0.5, 1, 0.1), colsample_bytree = c(0.5, 1, 0.1))
      parameters_df = expand.grid(max_depth = seq(3, 10, 1), eta = seq(0.1, 0.5, 0.1))
      lowest_error_list = list()
      
      # Hyperparameter tuning using grid search
      for (row in 1:nrow(parameters_df)){
        xgcv <- xgboost:::xgboost(data = full_train_df,
                                  label = train_y,
                                  booster = "gbtree",
                                  objective = "reg:linear",
                                  eval_metric = "rmse",
                                  max_depth = parameters_df$max_depth[row],
                                  eta = parameters_df$eta[row],
                                  nrounds = 250,
                                  early_stopping_rounds = 30,
                                  print_every_n = 10)
        
        lowest_error <- data.frame("min_error" = min(xgcv$evaluation_log$train_rmse))
        lowest_error_list[[row]] <- lowest_error
      }
      
      lowest_error_df <- do.call(rbind, lowest_error_list)
      grid_search_output <- cbind(lowest_error_df, parameters_df)
      optimised_parameters <- grid_search_output[grid_search_output$min_error == min(grid_search_output$min_error),]
      
      print(optimised_parameters)
      
      # Train a XGBoost model
      final_model <- xgboost:::xgboost(data = full_train_df,
                                       label = train_y,
                                       booster = "gbtree",
                                       objective = "reg:linear",
                                       eval_metric = "rmse",
                                       max_depth = optimised_parameters$max_depth,
                                       eta = optimised_parameters$eta,
                                       nround = 250,
                                       early_stopping_rounds = 30)
    }
    
    # Obtain predictions
    if(regression_type == "lasso")
      predictions <- predict(final_model, full_test_df)
    else if(regression_type == "linear")
      predictions <- predict.glm(final_model, as.data.frame(full_test_df))
    else if(regression_type == "xgboost")
      predictions <- predict(final_model, data.matrix(full_test_df))
    
    # Rescale predictions if required
    if(log_transformation)
      converted_predictions <- exp(predictions) - 1
    else
      converted_predictions <- predictions
    
    df_forecast[trainable_row_indexes, f] <- converted_predictions
  }
  
  # Replace forecasts if you want
  if(!is.null(forecasts_to_be_replaced) & !is.null(replacable_row_indexes)){
    df_forecast[replacable_row_indexes,] <- data.matrix(forecasts_to_be_replaced, rownames.force = NA)
  }
  
  df_forecast[df_forecast < 0] <- 0
  
  # Round final forecasts if required
  if(integer_conversion)
    df_forecast <- round(df_forecast)
  
  if(log_transformation)
    output_file_name <- paste0(output_file_name, "_log")
  
  write.table(df_forecast, file.path(FORECASTS_DIR, paste0(output_file_name, "_", regression_type, "_per_horizon_forecasts.txt"), fsep = "/"), row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  # Calculate errors
  print("Calculating errors")
  calculate_errors(df_forecast, test_set, training_set, seasonality, file.path(ERRORS_DIR, paste0(output_file_name, "_", regression_type, "_per_horizon"), fsep = "/"), address_near_zero_instability)
}


# Sample Usage

# NN5 Weekly
# Lasso Regression
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)

# Linear Regression
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")

# XGBoost
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt"), 8, "nn5_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")


# Ausgrid Weekly
# Lasso Regression
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE)

# Linear Regression
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "linear")

# XGBoost
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")
run_single_model_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "ausgrid_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "ausgrid_weekly_results.txt"), 8, "ausgrid_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, regression_type = "xgboost")


#Kaggle Web Traffic Weekly
# Lasso Regression
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)

# Linear Regression
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "linear")
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "linear")
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "linear")

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "linear")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "linear")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "linear")

# XGBoost
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "xgboost")
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "xgboost")
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "xgboost")

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "xgboost")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "xgboost")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_weekly_results.txt"), 8, "kaggle_web_traffic_weekly", optimal_k_value = 5, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE, regression_type = "xgboost")


#San Fransico Traffic Weekly
# Lasso Regression
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, address_near_zero_instability = TRUE, feature_freqency = 1)
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1)
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE)

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE)

# Linear Regression
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1, regression_type = "linear")
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1, regression_type = "linear")
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, regression_type = "linear")

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1, regression_type = "linear")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1, regression_type = "linear")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, regression_type = "linear")

# XGBoost
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1, regression_type = "xgboost")
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1, regression_type = "xgboost")
run_single_model_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, regression_type = "xgboost")

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1, regression_type = "xgboost")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, feature_freqency = 1, regression_type = "xgboost")
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "traffic_weekly_dataset.txt"), file.path(BASE_DIR, "datasets", "traffic_weekly_results.txt"), 8, "traffic_weekly", optimal_k_value = 15, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, regression_type = "xgboost")


# Daily datasets
# NN5 Daily
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_results.txt"), 56, "nn5_daily", optimal_k_value = 1, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, address_near_zero_instability = TRUE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_results.txt"), 56, "nn5_daily", optimal_k_value = 1, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "nn5_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_results.txt"), 56, "nn5_daily", optimal_k_value = 1, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE)

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_results.txt"), 56, "nn5_daily", optimal_k_value = 1, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_results.txt"), 56, "nn5_daily", optimal_k_value = 1, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "nn5_dataset.txt"), file.path(BASE_DIR, "datasets", "nn5_results.txt"), 56, "nn5_daily", optimal_k_value = 1, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE)


#Kaggle Web Traffic Daily
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_results.txt"), 59, "kaggle_web_traffic_daily", optimal_k_value = 1, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = TRUE, write_sub_model_forecasts = TRUE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_results.txt"), 59, "kaggle_web_traffic_daily", optimal_k_value = 1, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_single_model_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_results.txt"), 59, "kaggle_web_traffic_daily", optimal_k_value = 1, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)

run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_results.txt"), 59, "kaggle_web_traffic_daily", optimal_k_value = 1, log_transformation = TRUE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_results.txt"), 59, "kaggle_web_traffic_daily", optimal_k_value = 1, log_transformation = FALSE, use_features = TRUE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)
run_per_horizon_regression(file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_dataset.txt"), file.path(BASE_DIR, "datasets", "kaggle_web_traffic_1000_results.txt"), 59, "kaggle_web_traffic_daily", optimal_k_value = 1, log_transformation = TRUE, use_features = FALSE, calculate_sub_model_forecasts = FALSE, write_sub_model_forecasts = FALSE, address_near_zero_instability = TRUE, integer_conversion = TRUE)


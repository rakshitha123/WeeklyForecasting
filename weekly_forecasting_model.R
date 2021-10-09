library(smooth)

set.seed(123)


# DEFINE PATHS AND OTHER CONSTANT VARIABLES

BASE_DIR <- "WeeklyForecasting"
FORECASTS_DIR <- file.path(BASE_DIR, "results", "forecasts", fsep = "/")
ERRORS_DIR <- file.path(BASE_DIR, "results", "errors", fsep = "/")
SEASONALITY <- 365.25/7 
GLOBAL_MODELs <- c("rnn")
LOCAL_MODELS <- c("dhr_arima", "tbats", "theta")
METHODS <- c(LOCAL_MODELS, GLOBAL_MODELs) 
LAMBDAS_TO_TRY <- 10^seq(-3, 5, length.out = 100)
PHASES <- c("validation", "test")


######################################################################################################


# CHANGE THE FOLLOWING PARAMETERS ACCORDINGLY TO RUN THE MODEL WITH ANY DATASET

training_set_path <- file.path(BASE_DIR, "datasets", "nn5_weekly_dataset.txt") # The file path of the training dataset
test_set_path <- file.path(BASE_DIR, "datasets", "nn5_weekly_results.txt")     # The file path of the actual results corresponding with the expected forecasts. This will be used for final error calculation
forecast_horizon <- 8                                                          # The expected forecast horizon
dataset_name <- "nn5_weekly"                                                   # A string value for the dataset name. This will be used to as the prefix of the forecasts/error files
optimal_k_value <- 5                                                           # The k value that should be used when running dynamic harmonic regression arima (dhr_arima) model
calculate_sub_model_forecasts <- TRUE                                          # Whether the sub-models forecasts should be calculated or not. If you have already calculated the sub-model forecasts set this as FALSE and place the forecast files inside the FORECASTS_DIR with the name <dataset_name>_<method_name>_<phase>_forecasts.txt
write_sub_model_forecasts <- TRUE                                              # Whether the sub-model forecasts should be separately written into files. This will be useful when calculating the sub-model forecasts for the first time
integer_conversion <- FALSE                                                    # Whether the forecasts should be rounded or not
address_near_zero_instability <- FALSE                                         # Whether the results can have zeros or not. This will be used when calculating sMAPE
seasonality <- SEASONALITY                                                     # Seasonality that should be used to calculate MASE



######################################################################################################


# FORECASTING FUNCTIONS

# Calculate theta forecasts
get_theta_forecasts <-function(time_series, forecast_horizon){
  tryCatch(
    forecast:::thetaf(y = time_series, h = forecast_horizon)$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate tbats forecasts
get_tbats_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::tbats(time_series), h = forecast_horizon)$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate dynamic harmonic regression arima forecasts
get_dhr_arima_forecasts <- function(time_series, forecast_horizon, optimal_k_value){
  tryCatch({
    xreg <- forecast:::fourier(time_series, K = optimal_k_value)
    model <- forecast:::auto.arima(time_series, xreg = xreg, seasonal = FALSE)
    xreg1 <- forecast:::fourier(time_series, K = optimal_k_value, h = forecast_horizon)
    forecast(model, xreg = xreg1)$mean
  }, error = function(e) {   
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Retrieve the RNN forecasts corresponding with the given series identified by index
get_rnn_forecasts <- function(forecast_path, index){
  tryCatch({
    forecasts <- read.csv(forecast_path, header = FALSE, sep = " ")
    as.numeric(forecasts[index,])
  }, error = function(e) {   
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


######################################################################################################


# ERROR CALCULATION FUNCTIONS


# Function to calculate series wise smape values
#
# Parameters
# forecasts - A matrix containing forecasts for a set of series
#             no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon 
# test_set - A matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
# address_near_zero_instability - Whether the forecasts or actual values contain zeros or not
calculate_smape <- function(forecasts, test_set, address_near_zero_instability = FALSE){
  if (address_near_zero_instability == 1) {
    # define the custom smape function
    epsilon <- 0.1
    sum <- NULL
    comparator <- data.frame(matrix((0.5 + epsilon), nrow = nrow(test_set), ncol = ncol(test_set)))
    sum <- pmax(comparator, (abs(forecasts) + abs(test_set) + epsilon))
    smape <- 2 * abs(forecasts - test_set) / (sum)
  }else {
    smape <- 2 * abs(forecasts - test_set) / (abs(forecasts) + abs(test_set))
  }
  
  smape_per_series <- rowMeans(smape, na.rm = TRUE)
  
  smape_per_series
}


# Function to calculate series wise mase values
#
# Parameters
# forecasts - A matrix containing forecasts for a set of series
#             no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon 
# test_set - A matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
# training_set - A list containing the training series
# seasonality - Frequency of the dataset, e.g. 365.25/7 for weekly
calculate_mase <- function(forecasts, test_set, training_set, seasonality){
  mase_per_series <- NULL
  
  for(k in 1 :nrow(forecasts)){
    mase <- MASE(as.numeric(test_set[k,]), as.numeric(forecasts[k,]), mean(abs(diff(as.numeric(unlist(training_set[k], use.names = FALSE)), lag = seasonality, differences = 1))))
    
    if(is.na(mase)){
      mase <- MASE(as.numeric(test_set[k,]), as.numeric(forecasts[k,]), mean(abs(diff(as.numeric(unlist(training_set[k], use.names = FALSE)), lag = 1, differences = 1))))
    }
    
    mase_per_series[k] <- mase
  }
  
  mase_per_series
}


# Function to provide a summary of the 2 error metrics: smape and mase
#
# Parameters
# forecasts - A matrix containing forecasts for a set of series
#             no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon 
# test_set - A matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
# training_set - A list containing the training series
# seasonality - Frequency of the dataset
# output_file_name - The prefix of error file names
# address_near_zero_instability - Whether the forecasts or actual values can have zeros or not
calculate_errors <- function(forecasts, test_set, training_set, seasonality, output_file_name, address_near_zero_instability = FALSE){
  # Calculating smape
  smape_per_series <- calculate_smape(forecasts, test_set, address_near_zero_instability)
  
  # Calculating mase
  mase_per_series <- calculate_mase(forecasts, test_set, training_set, seasonality)
  
  mean_smape <- paste0("Mean SMAPE: ", mean(smape_per_series))
  median_smape <- paste0("Median SMAPE: ", median(smape_per_series))
  mean_mase <- paste0("Mean MASE: ", mean(mase_per_series))
  median_mase <- paste0("Median MASE: ", median(mase_per_series))
  
  print(mean_smape)
  print(median_smape)
  print(mean_mase)
  print(median_mase)
  
  # Writing error measures into files
  write.table(smape_per_series, paste0(output_file_name, "_smape.txt"), row.names = FALSE, col.names = FALSE, sep = ",", quote = FALSE)
  write.table(mase_per_series, paste0(output_file_name, "_mase.txt"), row.names = FALSE, col.names = FALSE, sep = ",", quote = FALSE)
  write(c(mean_smape, median_smape, mean_mase, median_mase, "\n"), file = paste0(output_file_name, ".txt"), append = FALSE)
}



######################################################################################################

# OTHER HELPER FUNCTIONS REQUIRED FOR THE WEEKLY FORECASTING MODEL


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



######################################################################################################



# EXECUTION OF THE PROPOSED WEEKLY FORECASTING MODEL


output_file_name <- dataset_name
  
print("Loading data")
  
# Obtain traning and test sets
training_test_sets <- create_traing_test_sets(training_set_path, test_set_path)
training_set <- training_test_sets[[1]]
test_set <- training_test_sets[[2]]
  
df_forecast <- matrix(nrow = length(training_set), ncol = forecast_horizon)
trainable_row_indexes <- 1:length(training_set)
  
full_train_df <- NULL
full_test_df <- NULL
  
start_time <- Sys.time()
  
# Calculate sub-model forecasts if required. Otherwise, load the pre-calculated forecasts
if(calculate_sub_model_forecasts){
  print("Started calculating sub-model forecasts")
  all_forecast_matrices <- do_forecasting(training_set, forecast_horizon, dataset_name, optimal_k_value, integer_conversion, write_sub_model_forecasts)
  print("Finished calculating sub-model forecasts")
}else
  all_forecast_matrices <- load_forecasts(training_set, forecast_horizon, PHASES, METHODS, file.path(FORECASTS_DIR, dataset_name, fsep = "/"))
  
  
for(method in METHODS){
  validation_forecasts <- eval(parse(text = paste0("all_forecast_matrices[['", method, "_validation_forecasts']]")))
  test_forecasts <- eval(parse(text = paste0("all_forecast_matrices[['", method, "_test_forecasts']]")))
    
  # Transform forecasts into log scale 
  validation_forecasts <- log(validation_forecasts + 1)
  test_forecasts <- log(test_forecasts + 1)
    
  # Aggregate forecasts with the training and test dataframes after converting them into vectors
  full_train_df <- cbind(full_train_df, create_vector(validation_forecasts[trainable_row_indexes,]))
  full_test_df <- cbind(full_test_df, create_vector(test_forecasts[trainable_row_indexes,]))
}
  
# Modify column names of train and test dataframes
colnames(full_train_df) <- sprintf("X%s",seq(1:length(METHODS)))
colnames(full_test_df) <- sprintf("X%s",seq(1:length(METHODS)))
  
full_train_df <- as.matrix(full_train_df)
full_test_df <- as.matrix(full_test_df)
  
train_forecasts <- all_forecast_matrices[["training_forecasts"]]
train_y <- create_vector(train_forecasts[trainable_row_indexes,])
  
output_file_name <- paste0(output_file_name, "_log")
train_y <- log(train_y + 1)
train_y <- as.matrix(train_y)
colnames(train_y) <- "y"
  
print("Processing ensemble model")
  
# Find the best lambda value to be used with 10-fold cross validation
lasso_cv <- glmnet:::cv.glmnet(full_train_df, train_y, alpha = 1, lambda = LAMBDAS_TO_TRY, standardize = TRUE, nfolds = 10, lower.limits = rep(0, ncol(full_train_df)), upper.limits = rep(1, ncol(full_train_df)))   
lambda_cv <- lasso_cv$lambda.min
print(paste0("Optimised lambda = ", lambda_cv))
  
# Train a lasso regression model
final_model <- glmnet:::glmnet(full_train_df, train_y, alpha = 1, lambda = lambda_cv, standardize = TRUE, lower.limits = rep(0, ncol(full_train_df)), upper.limits = rep(1, ncol(full_train_df)))
print(coef(final_model))

  
# Obtain predictions
predictions <- predict(final_model, full_test_df)
 
# Rescale predictions 
converted_predictions <- exp(predictions) - 1
  
df_forecast[trainable_row_indexes,] <- matrix(converted_predictions, ncol = forecast_horizon, byrow = FALSE)
df_forecast[df_forecast < 0] <- 0
  
# Round final forecasts if required
if(integer_conversion)
  df_forecast <- round(df_forecast)
  
write.table(df_forecast, file.path(FORECASTS_DIR, paste0(output_file_name, "_lasso_single_model_forecasts.txt"), fsep = "/"), row.names = FALSE, col.names = FALSE, quote = FALSE)
  
end_time <- Sys.time()
  
# Execution time
exec_time <- end_time - start_time
print(paste0("Execution time: ", exec_time, " ", attr(exec_time, "units")))
  
# Calculate errors
print("Calculating errors")
calculate_errors(df_forecast, test_set, training_set, seasonality, file.path(ERRORS_DIR, paste0(output_file_name, "_lasso_single_model"), fsep = "/"), address_near_zero_instability)


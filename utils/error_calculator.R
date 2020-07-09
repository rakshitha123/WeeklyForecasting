library(smooth)

# Functions to calculate smape and mase


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

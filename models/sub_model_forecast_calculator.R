# Implementations of a set of forecasting models
#
# Each function takes 2 parameters
# time_series - A ts object representing the time series that should be used with model training
# forecast_horizon - Expected forecast horizon
#
# The get_dhr_arima_forecasts function additionally uses optimal_k_value parameter which requires the k value that should be used during model training
#
# If a model fails to provide forecasts, it will return snaive forecasts


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
    forecasts <- read.csv(forecast_path, header = FALSE)
    as.numeric(forecasts[index,])
  }, error = function(e) {   
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate snaive forecasts
get_snaive_forecasts <- function(time_series, forecast_horizon){
  forecast:::snaive(time_series, h = forecast_horizon)$mean
}

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
    forecasts <- read.csv(forecast_path, header = FALSE, sep = " ")
    as.numeric(forecasts[index,])
  }, error = function(e) {   
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate holt forecasts
get_holt_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::holt(time_series, h = forecast_horizon))$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate simple exponential smoothing forecasts
get_ses_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::ses(time_series, h = forecast_horizon))$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate simple exponential smoothing forecasts
get_es_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    smooth:::es(time_series, initial = "backcasting", h = forecast_horizon)$forecast
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate croston forecasts
get_croston_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::croston(time_series, h = forecast_horizon))$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate ets forecasts
get_ets_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::ets(time_series, model = "AZN"), h = forecast_horizon)$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate arima forecasts
get_arima_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast::auto.arima(time_series, stepwise = FALSE, approximation = FALSE), h = forecast_horizon)$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate nnetar forecasts
get_nnetar_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast::nnetar(time_series), h = forecast_horizon)$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate stlmar forecasts
get_stlmar_forecasts <- function(time_series, forecast_horizon){
  model <- tryCatch({
    forecast::stlm(time_series, modelfunction = stats::ar)
  }, error = function(e) forecast::auto.arima(time_series, d = 0, D = 0))
  forecast::forecast(model, h = forecast_horizon)$mean
}


# Calculate rw_drift forecasts
get_rw_drift_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast::rwf(time_series, drift = TRUE, h = length(time_series)), h = forecast_horizon)$mean
    , error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
}


# Calculate naive forecasts
get_naive_forecasts <- function(time_series, forecast_horizon){
  forecast:::naive(time_series, h = forecast_horizon)$mean
}


# Calculate snaive forecasts
get_snaive_forecasts <- function(time_series, forecast_horizon){
  forecast:::snaive(time_series, h = forecast_horizon)$mean
}





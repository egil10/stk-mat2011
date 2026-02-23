# Install the arrow package for Parquet support if you don't have it
# install.packages("arrow")

library(tidyquant) 
library(tidyverse)
library(arrow)

# 1. Define assets and timeframe
tickers <- c("AAPL", "MSFT", "NVDA", "AMZN", "META", 
             "GOOGL", "TSLA", "AVGO", "WMT", "JPM", 
             "XOM", "V", "JNJ", "MA", "COST")

end_date   <- today()
start_date <- end_date - years(2)

# 2. Fetch the raw pricing data
raw_data <- tq_get(tickers, 
                   get  = "stock.prices", 
                   from = start_date, 
                   to   = end_date)

# 3. Pivot into a Price Matrix
price_matrix <- raw_data %>%
  select(symbol, date, adjusted) %>%
  pivot_wider(names_from = symbol, values_from = adjusted) %>%
  drop_na()

# 4. Calculate Daily Returns Matrix
returns_matrix <- raw_data %>%
  group_by(symbol) %>%
  tq_transmute(select     = adjusted, 
               mutate_fun = periodReturn, 
               period     = "daily", 
               col_rename = "daily_return") %>%
  pivot_wider(names_from = symbol, values_from = daily_return) %>%
  drop_na()

# 5. --- Setup the Save Directory (processed data) ---
data_dir <- "../data/processed"
if (!dir.exists(data_dir)) {
  dir.create(data_dir, recursive = TRUE)
}

# 6. --- Save the Data ---

# Option A: Save as Parquet (Highly recommended for Python handoff)
write_parquet(price_matrix, file.path(data_dir, "15_asset_prices.parquet"))
write_parquet(returns_matrix, file.path(data_dir, "15_asset_returns.parquet"))

# Option B: Save as CSV (Good to have for quick viewing in Excel/Numbers)
write_csv(price_matrix, file.path(data_dir, "15_asset_prices.csv"))
write_csv(returns_matrix, file.path(data_dir, "15_asset_returns.csv"))

print("Data successfully saved to ../data/processed!")


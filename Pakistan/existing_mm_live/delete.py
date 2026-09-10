
import pandas as pd
d='/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/diagnostics/'
pd.read_csv(d+'fill_prob_orders.csv').to_parquet(d+'fill_prob_orders.parquet')
print('done')

from pathlib import Path
import duckdb
import pandas as pd
from datetime import datetime

# Force pandas to show all rows and columns without wrapping or truncating (no dots)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

IN_DIR = Path("/Users/shazzak/Library/CloudStorage/"
              "GoogleDrive-shazzak@gmail.com/My Drive/Capital Stake - Parsed")

tables = ["trades", "misc", "ob_updates", "ob_snapshot"]

# Get current date and time, format it as YYYY-MM-DD_HH-MM-SS
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the output text file with the timestamp appended
output_file = f"schema_output_{timestamp}.txt"

# Open the file in write mode
with open(output_file, "w") as f:
    for table in tables:
        print(f"Processing {table}...")  # Prints to terminal so you know it's working

        # Write the table name header to the file
        f.write(f"========== Table: {table} ==========\n")

        try:
            # Run the query and convert to a Pandas DataFrame
            query = f"DESCRIBE SELECT * FROM read_parquet('{IN_DIR}/{table}/date=2026-06-30/*.parquet')"
            df_schema = duckdb.sql(query).df()

            # Write the dataframe to the text file using .to_string()
            f.write(df_schema.to_string())
            f.write("\n\n")  # Add spacing between tables

        except Exception as e:
            f.write(f"Error reading {table}: {e}\n\n")

print(f"\nDone! You can now find and upload '{output_file}' from your project folder.")
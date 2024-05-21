import pandas as pd

# Read Parquet file
df = pd.read_parquet('00001.parquet')
df.to_csv('00001.csv', index=False)

# df = pd.read_parquet('part-00000-tid-7242577580351498371-62730500-d341-4dc9-a022-896b3d29acac-23132-1-c000.snappy.parquet')
# df.to_csv('part-00000-tid-7242577580351498371-62730500-d341-4dc9-a022-896b3d29acac-23132-1-c000.snappy.csv', index=False)
# Save DataFrame as CSV
# df = pd.read_json('./tiktok/tiktok.json')

# df.to_csv('./tiktok/2.csv', index=False)


# import os
# import pandas as pd

# # Define the directory paths
# parquet_folder = './jsons/'
# csv_folder = './csvs/'

# # Create the CSV folder if it doesn't exist
# if not os.path.exists(csv_folder):
#     os.makedirs(csv_folder)

# # Initialize index for numbering CSV files
# csv_index = 1

# # Iterate over all files in the Parquet folder
# for filename in os.listdir(parquet_folder):
#     if filename.endswith('.json'):
#         # Read the Parquet file into a DataFrame
#         df = pd.read_json(os.path.join(parquet_folder, filename))
        
#         # Define the CSV filename with consecutive numbering
#         csv_filename = os.path.join(csv_folder, f'{csv_index}.csv')
        
#         # Save the DataFrame as a CSV file
#         df.to_csv(csv_filename, index=False)
        
#         print(f"Saved {filename} as {csv_filename}")
        
#         # Increment the index for the next CSV file
#         csv_index += 1

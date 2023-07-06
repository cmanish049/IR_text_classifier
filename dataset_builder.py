import os
import csv
import pandas as pd
# Data Handling and Processing

business_path = 'bbc/business/'
sports_path = 'bbc/sport/'
health_path = 'bbc/health/'
# will be loaded from news_df.csv

output_file = 'news_df.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(["Text", "Class"])

# Function to read the text files into one dataframe
def readfiles_to_dataframe(directory, category):
    output_file = 'news_df.csv'
    arr = os.listdir(directory)
    arr.sort()
    # Open the output CSV file in append mode
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Iterate over the files in the directory
        for filename in arr:
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)

                # Open each text file and read its content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Append the content as a row in the CSV file
                writer.writerow([content, str(category)])

readfiles_to_dataframe(business_path, 'business')
readfiles_to_dataframe(sports_path, 'sport')
readfiles_to_dataframe(health_path, 'health')
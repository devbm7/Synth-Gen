import json
import csv

# Define the path to the input JSON file
json_file_path = 'data/data.json'

# Read JSON data from the file
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Extract the list of data entries from the JSON data
data_entries = []
for entry in json_data:
    data_entries.extend(entry['data']['data'])
# Define the CSV file path
csv_file_path = 'data/data.csv'

# Write data to CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=data_entries[0].keys())
    writer.writeheader()
    writer.writerows(data_entries)

print(f"Data successfully written to {csv_file_path}")

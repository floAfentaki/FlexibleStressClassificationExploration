import os
import re
import pandas as pd

# Set the base directory
base_dir = os.getcwd()

# Define the paths to be scanned for each method
# data_paths = {
#     "disr": [
#         "DT/wesad/disr/float_models_4_synth",
#         "DT/wesad/disr/float_models_6_synth",
#         "DT/wesad/disr/float_models_8_synth",
#         "DT/wesad/disr/float_models_10_synth",
#     ],
#     "fisher_score": [
#         "DT/wesad/fisher_score/float_models_4_synth",
#         "DT/wesad/fisher_score/float_models_6_synth",
#         "DT/wesad/fisher_score/float_models_8_synth",
#         "DT/wesad/fisher_score/float_models_10_synth",
#     ],
#     "jmi": [
#         "DT/wesad/jmi/float_models_4_synth",
#         "DT/wesad/jmi/float_models_6_synth",
#         "DT/wesad/jmi/float_models_8_synth",
#         "DT/wesad/jmi/float_models_10_synth",
#     ]
# }


# data_paths = {
#     "disr": [
#         "SVM/wesad/disr/float_models_4_synth",
#         "SVM/wesad/disr/float_models_6_synth",
#         "SVM/wesad/disr/float_models_8_synth",
#         "SVM/wesad/disr/float_models_10_synth",
#     ],
#     "fisher_score": [
#         "SVM/wesad/fisher_score/float_models_4_synth",
#         "SVM/wesad/fisher_score/float_models_6_synth",
#         "SVM/wesad/fisher_score/float_models_8_synth",
#         "SVM/wesad/fisher_score/float_models_10_synth",
#     ],
#     "jmi": [
#         "SVM/wesad/jmi/float_models_4_synth",
#         "SVM/wesad/jmi/float_models_6_synth",
#         "SVM/wesad/jmi/float_models_8_synth",
#         "SVM/wesad/jmi/float_models_10_synth",
#     ]
# }

data_paths = {
    "disr": [
        # "SVM/wesad/disr/float_models_4_synth",
        # "SVM/wesad/disr/float_models_6_synth",
        "MLP/wesad/disr/float_models_8_synth",
        # "SVM/wesad/disr/float_models_10_synth",
    ],
    "fisher_score": [
        # "SVM/wesad/fisher_score/float_models_4_synth",
        # "SVM/wesad/fisher_score/float_models_6_synth",
        "MLP/wesad/fisher_score/float_models_8_synth",
        # "SVM/wesad/fisher_score/float_models_10_synth",
    ],
    "jmi": [
        # "SVM/wesad/jmi/float_models_4_synth",
        # "SVM/wesad/jmi/float_models_6_synth",
        "MLP/wesad/jmi/float_models_8_synth",
        # "SVM/wesad/jmi/float_models_10_synth",
    ]
}


data_paths = {
    "disr": [
        # "SVM/wesad/disr/float_models_4_synth",
        # "SVM/wesad/disr/float_models_6_synth",
        "MLP/wesad/disr/float_models_8_synth",
        # "SVM/wesad/disr/float_models_10_synth",
    ],
    "fisher_score": [
        # "SVM/wesad/fisher_score/float_models_4_synth",
        # "SVM/wesad/fisher_score/float_models_6_synth",
        "MLP/wesad/fisher_score/float_models_8_synth",
        # "SVM/wesad/fisher_score/float_models_10_synth",
    ],
    "jmi": [
        # "SVM/wesad/jmi/float_models_4_synth",
        # "SVM/wesad/jmi/float_models_6_synth",
        "MLP/wesad/jmi/float_models_8_synth",
        # "SVM/wesad/jmi/float_models_10_synth",
    ]
}


# Function to reliably extract bitwidth from the path using regex
def extract_bitwidth(path):
    match = re.search(r'float_models_(\d+)_synth', path)
    if match:
        return match.group(1)  # Extract the bitwidth number
    return 'unknown'

# List to hold the combined data
combined_data = []

# Process each method and collect the data
for method, paths in data_paths.items():
    for path in paths:
        file_path = os.path.join(base_dir, path, "csv.csv")
        bitwidth = extract_bitwidth(path)  # Extract bitwidth using regex

        # Check if the file exists and has a .csv extension
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            try:
                # Read the CSV with delimiter ';'
                df = pd.read_csv(file_path, delimiter=';')

                # Validate that the required columns are present
                required_columns = {'Number', 'Accuracy', 'Area', 'Power'}
                if required_columns.issubset(df.columns):
                    # Ensure that we aren't mistakenly adding malformed rows
                    df = df.dropna(subset=['Number', 'Accuracy', 'Area', 'Power'])

                    # Add the required columns to each row
                    df['Dataset'] = 'wesad'  # Add the dataset name
                    df['FeatureSelection'] = method  # Add the feature selection method (disr, fisher_score, jmi)
                    df['bitwidth'] = bitwidth  # Add bitwidth information from path

                    # Append the DataFrame to combined_data
                    combined_data.append(df[['Dataset', 'FeatureSelection', 'Number', 'Accuracy', 'Area', 'Power', 'bitwidth']])
                else:
                    print(f"Skipping file {file_path} - missing required columns: {required_columns - set(df.columns)}.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"Skipping file {file_path} - not a valid .csv file.")

# Concatenate all the data into a single DataFrame
df_combined = pd.concat(combined_data, ignore_index=True)

# Write the combined data to a CSV file
output_csv = os.path.join(base_dir, "modelMLP.csv")
df_combined.to_csv(output_csv, index=False)

print(f"Combined data written to {output_csv}")

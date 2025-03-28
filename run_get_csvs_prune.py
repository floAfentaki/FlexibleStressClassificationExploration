import os
import re
import pandas as pd

# Set the base directory
base_dir = os.getcwd()

# Define the paths to be scanned for each method
# data_paths = {
#     "disr": [      
#         "MLP/wesad/disr/pruning_ft_l2_4_synth",
#         "MLP/wesad/disr/pruning_ft_l2_6_synth",
#         "MLP/wesad/disr/pruning_ft_l2_8_synth",
#         "MLP/wesad/disr/pruning_ft_l2_10_synth",
#     ],
#     "fisher_score": [
#         "MLP/wesad/fisher_score/pruning_ft_l2_4_synth",
#         "MLP/wesad/fisher_score/pruning_ft_l2_6_synth",
#         "MLP/wesad/fisher_score/pruning_ft_l2_8_synth",
#         "MLP/wesad/fisher_score/pruning_ft_l2_10_synth",
#     ],
#     "jmi": [
#         "MLP/wesad/jmi/pruning_ft_l2_4_synth",
#         "MLP/wesad/jmi/pruning_ft_l2_6_synth",
#         "MLP/wesad/jmi/pruning_ft_l2_8_synth",
#         "MLP/wesad/jmi/pruning_ft_l2_10_synth",
#     ]
# }

data_paths = {
    "disr": [      
        "MLP/AffectiveROAD/disr/pruning_ft_l2_4_synth",
        "MLP/AffectiveROAD/disr/pruning_ft_l2_6_synth",
        "MLP/AffectiveROAD/disr/pruning_ft_l2_8_synth",
        "MLP/AffectiveROAD/disr/pruning_ft_l2_10_synth",
    ],
    "fisher_score": [
        "MLP/AffectiveROAD/fisher_score/pruning_ft_l2_4_synth",
        "MLP/AffectiveROAD/fisher_score/pruning_ft_l2_6_synth",
        "MLP/AffectiveROAD/fisher_score/pruning_ft_l2_8_synth",
        "MLP/AffectiveROAD/fisher_score/pruning_ft_l2_10_synth",
    ],
    "jmi": [
        "MLP/AffectiveROAD/jmi/pruning_ft_l2_4_synth",
        "MLP/AffectiveROAD/jmi/pruning_ft_l2_6_synth",
        "MLP/AffectiveROAD/jmi/pruning_ft_l2_8_synth",
        "MLP/AffectiveROAD/jmi/pruning_ft_l2_10_synth",
    ]
}

def extract_pruning(path):
    match = re.search(r'pruning_ft_([a-zA-Z0-9]+)_', path)
    if match:
        return match.group(1)  # Extract the pruning method (e.g., 'hessian')
    return 'unknown'


# Function to reliably extract bitwidth from the path using regex
def extract_bitwidth(path):
    match = re.search(r'_(\d+)_synth', path)
    if match:
        return match.group(1)  # Extract the bitwidth number
    return 'unknown'

# Function to extract sparsity rate from the 'Number' column
def extract_sparsity_rate(number_value):
    match = re.search(r'sparsity_(\d+_\d+)', number_value)
    if match:
        return match.group(1).replace('_', '.')  # Convert to '0.2', '0.5', etc.
    return 'unknown'

# List to hold the combined data
combined_data = []

# Process each method and collect the data
for FSmethod, paths in data_paths.items():
    for path in paths:
        file_path = os.path.join(base_dir, path, "csv.csv")
        bitwidth = extract_bitwidth(path)  # Extract bitwidth using regex
        PruneMethod=extract_pruning(path)
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

                    # Extract sparsity rate and clean up 'Number'
                    df['sparsity_rate'] = df['Number'].apply(extract_sparsity_rate)  # Extract sparsity rate
                    df['Number'] = '10'  # Set the Number column to '10'

                    # Add the required columns to each row
                    df['Dataset'] = 'AffectiveROAD'  # Add the dataset name
                    df['FeatureSelection'] = FSmethod  # Add the feature selection method (disr, fisher_score, jmi)
                    df['PruneMethod'] = PruneMethod  # Add the feature selection method (disr, fisher_score, jmi)
                    df['bitwidth'] = bitwidth  # Add bitwidth information from path

                    # Append the DataFrame to combined_data
                    combined_data.append(df[['Dataset', 'FeatureSelection', 'PruneMethod', 'Number', 'Accuracy', 'Area', 'Power', 'bitwidth', 'sparsity_rate']])
                else:
                    print(f"Skipping file {file_path} - missing required columns: {required_columns - set(df.columns)}.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"Skipping file {file_path} - not a valid .csv file.")

# Concatenate all the data into a single DataFrame
df_combined = pd.concat(combined_data, ignore_index=True)

# Write the combined data to a CSV file
# output_csv = os.path.join(base_dir, "model_quant_MLP.csv")
output_csv = os.path.join(base_dir, "model_quant_MLP_LBR_AffectiveROAD.csv")
# output_csv = os.path.join(base_dir, "model_Prune_MLP.csv")
# output_csv = os.path.join(base_dir, "model_ne_MLP.csv")
df_combined.to_csv(output_csv, index=False)

print(f"Combined data written to {output_csv}")

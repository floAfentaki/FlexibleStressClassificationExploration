import pandas as pd

# Load the dataset
csv_path = 'modelDT.csv'  # Replace with the path to your actual modelDT CSV file
# csv_path = 'modelSVM.csv'  # Replace with the path to your actual modelSVM CSV file
# csv_path = 'model_Prune_MLP.csv'
data = pd.read_csv(csv_path)

# Sort data by accuracy (descending) and area (ascending)
sorted_data = data.sort_values(by=['Accuracy', 'Area'], ascending=[False, True])

# Initialize an empty list to store Pareto points
pareto_points = []

# Start with the first point in the sorted list (highest accuracy and lowest area)
current_area = float('inf')  # Start with a very high value for area
for index, row in sorted_data.iterrows():
    if row['Area'] < current_area:
        pareto_points.append(row)
        current_area = row['Area']

# Convert the list of Pareto points to a DataFrame
pareto_df = pd.DataFrame(pareto_points)

# Display the Pareto points with all their statistics (including Power, Bitwidth)
print(pareto_df)

# Optionally save the Pareto points to a CSV file
# pareto_df.to_csv("ppDT.csv", index=False)

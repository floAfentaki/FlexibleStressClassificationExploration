import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec


def pareto_frontier_2d(Xs, Ys, maxX=False, maxY=True):
    # Convert pandas Series to lists
    Xs = Xs.tolist()
    Ys = Ys.tolist()
    
    # Sort the points by X (Area), ascending this time (minimize area)
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    
    # Pareto frontier starting with the first point in the sorted list
    p_front = [myList[0]]    
    for pair in myList[1:]:
        if maxY:
            if pair[1] > p_front[-1][1]:  # Keep the point if it has a higher Accuracy
                p_front.append(pair)
        else:
            if pair[1] < p_front[-1][1]:  # Keep the point if it has a lower Accuracy
                p_front.append(pair)
    
    # Extract the X and Y values into separate lists
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    
    return p_frontX, p_frontY

# Step 1: Load all CSV data
csv_file_dt = 'modelDT.csv'  # Replace with the path to your actual modelDT CSV file
csv_file_svm = 'modelSVM.csv'  # Replace with the path to your actual modelSVM CSV file
csv_file_mlp = 'model_quant_MLP.csv'  # Replace with the path to your actual modelMLP CSV file

data_dt = pd.read_csv(csv_file_dt)
data_svm = pd.read_csv(csv_file_svm)
data_mlp = pd.read_csv(csv_file_mlp)

data_dt['FeatureSelection'] = data_dt['FeatureSelection'].replace('fisher_score', 'fisher')
data_svm['FeatureSelection'] = data_svm['FeatureSelection'].replace('fisher_score', 'fisher')
data_mlp['FeatureSelection'] = data_mlp['FeatureSelection'].replace('fisher_score', 'fisher')

# Step 2: Filter the rows where bitwidth is 8 for all datasets
filtered_data_dt = data_dt
filtered_data_svm = data_svm
filtered_data_mlp = data_mlp

# Step 3: Create a folder 'fig' if it doesn't exist
output_folder = 'fig'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 4: Define a marker style dictionary for each FeatureSelection
# marker_styles = {
#     'disr': '.',          # Dot marker for 'disr'
#     'fisher': '*',  # Renamed from 'fisher_score' to 'fs' with star marker
#     'jmi': '+'            # Plus marker for 'jmi'
# }

marker_styles = {
    4: '.',  # Light pink
    6: '+', # Light blue
    8: '*', # Light green
    10:'x', # Light purple
}

# Define pastel colors from seaborn for the specific 'Number' values
pastel_colors = sns.color_palette("pastel", 6)
number_colors = {
    4: pastel_colors[0],  # Light pink
    6: pastel_colors[1], # Light blue
    8: pastel_colors[2], # Light green
    10: pastel_colors[3], # Light purple
}

# Define the color map function to assign pastel colors based on 'Number'
def color_map(number):
    return number_colors.get(number, 'lightgray')  # Lightgray for other numbers

# Step 5: Set up subplots (2 rows, 2 columns), with total width of 3.37 inches per subplot
fig = plt.figure(figsize=(3.37, 2))  # Total width: 3.37in per subplot, 2 rows
# fig, axes = plt.subplots(2, 2, figsize=(3.37, 2))  # Total width: 3.37in per subplot, 2 rows
gs = GridSpec(2, 2, figure=fig)  # First row: 1 unit high, second row: 2 units high
ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
ax1 = fig.add_subplot(gs[0, 1])  # First row, second column
ax2 = fig.add_subplot(gs[1, :])
# Step 6: Reduce space between subplots to maximize figure area
plt.subplots_adjust(left=0.09, right=0.9, top=0.9, bottom=0.12, wspace=0.25, hspace=0.25)

# Step 7: Plot for modelDT in the first subplot (ax0)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_dt[filtered_data_dt['bitwidth'] == feature_selection]
    colors = [color_map(n) for n in subset['bitwidth']]  # Apply color map to each Number
    ax0.scatter(
        subset['Area'], 
        subset['Accuracy'], 
        c=colors,                # Assign colors based on the Number column
        marker=marker,           # Assign different markers
        s=50,                    # Marker size (adjusted for visibility)
        alpha=0.8,               # Transparency for visibility
        label=feature_selection  # Add label for shared legend
    )

# Calculate ticks for x and y axes for DT plot
x_min_dt, x_max_dt = filtered_data_dt['Area'].min(), filtered_data_dt['Area'].max()
y_min_dt, y_max_dt = filtered_data_dt['Accuracy'].min(), filtered_data_dt['Accuracy'].max()
ax0.set_xticks(np.round(np.linspace(x_min_dt, x_max_dt, 3), 4))
ax0.set_yticks(np.round(np.linspace(y_min_dt, y_max_dt, 5)))
ax0.tick_params(axis='both', labelsize=7, pad=0.2)
ax0.text(0.7, 0.1, 'a) DT', ha='left', va='center', transform=ax0.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_dt['Area'], filtered_data_dt['Accuracy'], maxX=False, maxY=True)
ax0.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points


# Step 8: Plot for modelSVM in the second subplot (ax1)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_svm[filtered_data_svm['bitwidth'] == feature_selection]
    colors = [color_map(n) for n in subset['bitwidth']]  # Apply color map to each Number
    ax1.scatter(
        subset['Area'], 
        subset['Accuracy'], 
        c=colors,                # Assign colors based on the Number column
        marker=marker,           # Assign different markers
        s=50,                    # Marker size (adjusted for visibility)
        alpha=0.8,               # Transparency for visibility
        label=feature_selection  # Add label for shared legend
    )

# Calculate ticks for x and y axes for SVM plot
x_min_svm, x_max_svm = filtered_data_svm['Area'].min(), filtered_data_svm['Area'].max()
y_min_svm, y_max_svm = filtered_data_svm['Accuracy'].min(), filtered_data_svm['Accuracy'].max()
ax1.set_xticks(np.round(np.linspace(x_min_svm, x_max_svm, 3), 3))
ax1.set_yticks(np.round(np.linspace(y_min_svm, y_max_svm, 5)))
ax1.tick_params(axis='both', labelsize=7, pad=0.2)
ax1.text(0.7, 0.1, 'b) SVM', ha='left', va='center', transform=ax1.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_svm['Area'], filtered_data_svm['Accuracy'], maxX=False, maxY=True)
ax1.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points

# Step 9: Plot for modelMLP in the third subplot (ax2)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_mlp[filtered_data_mlp['bitwidth'] == feature_selection]
    colors = [color_map(n) for n in subset['bitwidth']]  # Apply color map to each Number
    ax2.scatter(
        subset['Area'], 
        subset['Accuracy'], 
        c=colors,                # Assign colors based on the Number column
        marker=marker,           # Assign different markers
        s=50,                    # Marker size (adjusted for visibility)
        alpha=0.8,               # Transparency for visibility
        label=feature_selection  # Add label for shared legend
    )

# Calculate ticks for x and y axes for MLP plot
x_min_mlp, x_max_mlp = filtered_data_mlp['Area'].min(), filtered_data_mlp['Area'].max()
y_min_mlp, y_max_mlp = filtered_data_mlp['Accuracy'].min(), filtered_data_mlp['Accuracy'].max()
ax2.set_xticks(np.round(np.linspace(x_min_mlp, x_max_mlp, 3), 3))
ax2.set_yticks(np.round(np.linspace(y_min_mlp, y_max_mlp, 5)))
ax2.tick_params(axis='both', labelsize=7, pad=0.2)
ax2.text(0.87, 0.1, 'c) MLP', ha='left', va='center', transform=ax2.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_mlp['Area'], filtered_data_mlp['Accuracy'], maxX=False, maxY=True)
ax2.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points


# Step 10: Remove individual y-axis labels and add a shared y-axis label for 'Accuracy'
fig.text(0.005, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=7)

# Step 11: Add a shared x-axis label for 'Area'
fig.text(0.5, 0.01, 'Area(cm^2)', ha='center', fontsize=7)
# axes[1, 1].tick_params(axis='both', labelsize=7, pad=0.2)

# Step 12: Create a combined legend with 3 rows and 6 columns
legend_handles = []
for number, color in number_colors.items():
    # for feature_selection, marker in marker_styles.items():
        handle = plt.Line2D([0], [0], marker=marker_styles[number], color=color, markersize=7, linestyle='None', label=f'{number}-bits')
        legend_handles.append(handle)

# Create the combined legend with 3 rows and 6 columns
fig.legend(
    handles=legend_handles, 
    loc='upper center', 
    fontsize=7, 
    ncol=10, 
    frameon=False, 
    bbox_to_anchor=(0.5, 1.025),
    handletextpad=-0.5,  # Reducing space between the symbol and text
    columnspacing=-0.02  # Reducing space between columns
)

# Step 13: Save the figure as a PDF in the 'fig' folder
output_pdf_path = os.path.join(output_folder, 'quant.pdf')
plt.savefig(output_pdf_path, format='pdf')

# Show the plot for visual confirmation (optional)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

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

TITLE_FONT="Nimbus Sans"
TITLE_WEIGHT="normal"

plt.rcParams['font.family'] = TITLE_FONT  # Correct font name
plt.rcParams['font.sans-serif'] = [TITLE_FONT]
plt.rc('pdf', fonttype=42)

# Step 1: Load all CSV data
csv_file_dt = 'modelDT.csv'  # Replace with the path to your actual modelDT CSV file
csv_file_svm = 'modelSVM.csv'  # Replace with the path to your actual modelSVM CSV file
csv_file_mlp = 'modelMLP.csv'  # Replace with the path to your actual modelMLP CSV file
# csv_file_mlp = 'model_quant_MLP_LBR_AffectiveROAD.csv'  # Replace with the path to your actual modelMLP CSV file

data_dt = pd.read_csv(csv_file_dt)
data_svm = pd.read_csv(csv_file_svm)
data_mlp = pd.read_csv(csv_file_mlp)


data_dt = data_dt[data_dt['Dataset'] == 'wesad']
data_svm = data_svm[data_svm['Dataset'] == 'wesad']
data_mlp = data_mlp[data_mlp['Dataset'] == 'wesad']
# print(data_dt.head())

data_dt['FeatureSelection'] = data_dt['FeatureSelection'].replace('fisher_score', 'fisher')
data_svm['FeatureSelection'] = data_svm['FeatureSelection'].replace('fisher_score', 'fisher')
data_mlp['FeatureSelection'] = data_mlp['FeatureSelection'].replace('fisher_score', 'fisher')

# Step 2: Filter the rows where bitwidth is 8 for all datasets
filtered_data_dt = data_dt[data_dt['bitwidth'] == 8]
filtered_data_svm = data_svm[data_svm['bitwidth'] == 8]
filtered_data_mlp = data_mlp[data_mlp['bitwidth'] == 8]

# Step 3: Create a folder 'fig' if it doesn't exist
output_folder = 'fig'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 4: Define a marker style dictionary for each FeatureSelection
marker_styles = {
    'disr': '.',          # Dot marker for 'disr'
    'fisher': '*',  # Renamed from 'fisher_score' to 'fs' with star marker
    'jmi': '+'            # Plus marker for 'jmi'
}

# Define pastel colors from seaborn for the specific 'Number' values
pastel_colors = sns.color_palette("pastel", 6)
number_colors = {
    5: pastel_colors[0],  # Light pink
    10: pastel_colors[1], # Light blue
    15: pastel_colors[2], # Light green
    20: pastel_colors[3], # Light purple
    25: pastel_colors[4], # Light orange
    30: pastel_colors[5], # Light yellow
}

# Define the color map function to assign pastel colors based on 'Number'
def color_map(number):
    return number_colors.get(number, 'lightgray')  # Lightgray for other numbers

# Step 5: Set up subplots (2 rows, 2 columns), with total width of 3.37 inches per subplot
fig = plt.figure(figsize=(3.37, 1.5))  # Total width: 3.37in per subplot, 2 rows

plt.subplots_adjust(left=0.1, right=0.96, top=0.82, bottom=0.17, wspace=0.3, hspace=0.5)
gs = GridSpec(2, 2, figure=fig)  # First row: 1 unit high, second row: 2 units high
ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
ax1 = fig.add_subplot(gs[0, 1])  # First row, second column
ax2 = fig.add_subplot(gs[1, :])

# Step 7: Plot for modelDT in the first subplot (ax0)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_dt[filtered_data_dt['FeatureSelection'] == feature_selection]
    colors = [color_map(n) for n in subset['Number']]  # Apply color map to each Number
    ax0.scatter(
        subset['Area'], 
        subset['Accuracy'], 
        c=colors,                # Assign colors based on the Number column
        marker=marker,           # Assign different markers
        s=50,                    # Marker size (adjusted for visibility)
        alpha=0.8,               # Transparency for visibility
        label=feature_selection  # Add label for shared legend
    )
    
# Calculate the greatest distance for the same number of features but different feature selection methods for DT
max_distance_dt = 0
for number in filtered_data_dt['Number'].unique():
    subset = filtered_data_dt[filtered_data_dt['Number'] == number]
    if len(subset) > 1:
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                distance = np.sqrt((subset.iloc[i]['Area'] - subset.iloc[j]['Area'])**2 + (subset.iloc[i]['Accuracy'] - subset.iloc[j]['Accuracy'])**2)
                if distance > max_distance_dt:
                    max_distance_dt = distance

print(f'Max distance for DT: {max_distance_dt}')

# Calculate the greatest distance for the same number of features but different feature selection methods for SVM
max_distance_svm = 0
for number in filtered_data_svm['Number'].unique():
    subset = filtered_data_svm[filtered_data_svm['Number'] == number]
    if len(subset) > 1:
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                distance = np.sqrt((subset.iloc[i]['Area'] - subset.iloc[j]['Area'])**2 + (subset.iloc[i]['Accuracy'] - subset.iloc[j]['Accuracy'])**2)
                if distance > max_distance_svm:
                    max_distance_svm = distance

print(f'Max distance for SVM: {max_distance_svm}')

# Calculate the greatest distance for the same number of features but different feature selection methods for MLP
max_distance_mlp = 0
for number in filtered_data_mlp['Number'].unique():
    subset = filtered_data_mlp[filtered_data_mlp['Number'] == number]
    if len(subset) > 1:
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                distance = np.sqrt((subset.iloc[i]['Area'] - subset.iloc[j]['Area'])**2 + (subset.iloc[i]['Accuracy'] - subset.iloc[j]['Accuracy'])**2)
                if distance > max_distance_mlp:
                    max_distance_mlp = distance

print(f'Max distance for MLP: {max_distance_mlp}')
# Calculate ticks for x and y axes for DT plot
x_min_dt, x_max_dt = filtered_data_dt['Area'].min(), filtered_data_dt['Area'].max()
y_min_dt, y_max_dt = filtered_data_dt['Accuracy'].min(), filtered_data_dt['Accuracy'].max()
ax0.set_xticks(np.round(np.linspace(x_min_dt, x_max_dt, 4), 3))
ax0.set_yticks(np.round(np.linspace(y_min_dt, y_max_dt, 4)))
ax0.set_ylim([y_min_dt-4, y_max_dt+4])  # Set y-axis limits
ax0.tick_params(axis='both', labelsize=7, pad=0.2)
ax0.text(0.75, 0.13, 'a) DT', ha='left', va='center', transform=ax0.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_dt['Area'], filtered_data_dt['Accuracy'], maxX=False, maxY=True)
ax0.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points

q_list=[]
fs_list=[]
for i in range(len(pareto_x)):
    fs_list.append(filtered_data_dt[(filtered_data_dt['Area'] == pareto_x[i]) & (filtered_data_dt['Accuracy'] == pareto_y[i])]['FeatureSelection'].values[0])

print(f'FS: {fs_list}')
total_count = len(fs_list)
disr_list = fs_list.count('disr')
fisher_list = fs_list.count('fisher')
jmi_list = fs_list.count('jmi')
print(f'DISR Percentage: {(disr_list / total_count) * 100:.2f}%')
print(f'Fisher Percentage: {(fisher_list / total_count) * 100:.2f}%')
print(f'JMI Percentage: {(jmi_list / total_count) * 100:.2f}%')
print('DT')

# Step 8: Plot for modelSVM in the second subplot (ax1)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_svm[filtered_data_svm['FeatureSelection'] == feature_selection]
    colors = [color_map(n) for n in subset['Number']]  # Apply color map to each Number
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
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xticks(np.round(np.linspace(x_min_svm, x_max_svm, 3), 3))
ax1.set_yticks(np.round(np.linspace(y_min_svm, y_max_svm, 4)))
ax1.set_ylim([y_min_svm-4, y_max_svm+4])  # Set y-axis limits
ax1.tick_params(axis='both', labelsize=7, pad=0.2)
ax1.text(0.7, 0.13, 'b) SVM', ha='left', va='center', transform=ax1.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_svm['Area'], filtered_data_svm['Accuracy'], maxX=False, maxY=True)
ax1.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points


q_list=[]
fs_list=[]
for i in range(len(pareto_x)):
    fs_list.append(filtered_data_svm[(filtered_data_svm['Area'] == pareto_x[i]) & (filtered_data_svm['Accuracy'] == pareto_y[i])]['FeatureSelection'].values[0])
  
print(f'FS: {fs_list}')
total_count = len(fs_list)
disr_list = fs_list.count('disr')
fisher_list = fs_list.count('fisher')
jmi_list = fs_list.count('jmi')
print(f'DISR Percentage: {(disr_list / total_count) * 100:.2f}%')
print(f'Fisher Percentage: {(fisher_list / total_count) * 100:.2f}%')
print(f'JMI Percentage: {(jmi_list / total_count) * 100:.2f}%')
print('SVM')


# Step 9: Plot for modelMLP in the third subplot (ax2)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_mlp[filtered_data_mlp['FeatureSelection'] == feature_selection]
    colors = [color_map(n) for n in subset['Number']]  # Apply color map to each Number
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
ax2.set_yticks(np.round(np.linspace(y_min_mlp, y_max_mlp, 4)))
ax2.set_ylim([y_min_mlp-4, y_max_mlp+4])  # Set y-axis limits
ax2.tick_params(axis='both', labelsize=7, pad=0.2)
ax2.text(0.85, 0.13, 'c) MLP', ha='left', va='center', transform=ax2.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_mlp['Area'], filtered_data_mlp['Accuracy'], maxX=False, maxY=True)
ax2.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points

q_list=[]
fs_list=[]
for i in range(len(pareto_x)):

    fs_list.append(filtered_data_mlp[(filtered_data_mlp['Area'] == pareto_x[i]) & (filtered_data_mlp['Accuracy'] == pareto_y[i])]['FeatureSelection'].values[0])

print(f'FS: {fs_list}')
total_count = len(fs_list)
disr_list = fs_list.count('disr')
fisher_list = fs_list.count('fisher')
jmi_list = fs_list.count('jmi')
print(f'DISR Percentage: {(disr_list / total_count) * 100:.2f}%')
print(f'Fisher Percentage: {(fisher_list / total_count) * 100:.2f}%')
print(f'JMI Percentage: {(jmi_list / total_count) * 100:.2f}%')
print('MLP')


# Step 10: Remove individual y-axis labels and add a shared y-axis label for 'Accuracy'
fig.text(0.005, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=7)


fig.text(0.5, 0.01, 'Area ($cm^2$)', ha='center', fontsize=7)
# fig.text(0.5, 0.01, 'Power ($mW$)', ha='center', fontsize=7)
ax1.tick_params(axis='both', labelsize=7, pad=0.2)



# Show the plot for visual confirmation (optional)
plt.show()


def make_plot():
    # Define a closure function to register as a callback
    def convert_ax_f_to_celsius(ax_f):
        """
        Update second axis according to first axis.
        """
        x1, x2 = ax_f.get_xlim()  # Get limits from first axis
        ax_c.set_xlim(fahrenheit2celsius(x1), fahrenheit2celsius(x2))  # Convert and set limits for second axis
        ax_c.figure.canvas.draw()

    fig, ax_f = plt.subplots()
    ax_c = ax_f.twiny()  # Use twiny() instead of twinx() to share the x-axis

    # Automatically update xlim of ax2 when xlim of ax1 changes
    ax_f.callbacks.connect("xlim_changed", convert_ax_f_to_celsius)

    ax_f.plot(np.linspace(-40, 120, 100), np.sin(np.linspace(-40, 120, 100)))  # Example plot
    ax_f.set_xlim(0, 100)

    ax_f.set_title('Two scales: Fahrenheit and Celsius (on X-axis)')
    ax_f.set_xlabel('Fahrenheit')
    ax_c.set_xlabel('Celsius')

    plt.show()

make_plot()

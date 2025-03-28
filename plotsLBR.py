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
# csv_file_mlp = 'model_quant_MLP.csv'  # Replace with the path to your actual modelMLP CSV file
# csv_file_mlp = 'model_quant_MLP_LBR_AffectiveROAD.csv'  # Replace with the path to your actual modelMLP CSV file
csv_file_mlp = 'model_quant_MLP_LBR.csv'  # Replace with the path to your actual modelMLP CSV file

data_dt = pd.read_csv(csv_file_dt)
data_svm = pd.read_csv(csv_file_svm)
data_dt = data_dt[data_dt['Dataset'] == 'wesad']
data_svm = data_svm[data_svm['Dataset'] == 'wesad']
data_mlp = pd.read_csv(csv_file_mlp)

data_dt['FeatureSelection'] = data_dt['FeatureSelection'].replace('fisher_score', 'fisher')
data_svm['FeatureSelection'] = data_svm['FeatureSelection'].replace('fisher_score', 'fisher')
data_mlp['FeatureSelection'] = data_mlp['FeatureSelection'].replace('fisher_score', 'fisher')

# Step 2: Filter the rows where bitwidth is 8 for all datasets
# filtered_data_dt = data_dt[data_dt['bitwidth'] == 8]
# filtered_data_svm = data_svm[data_svm['bitwidth'] == 8]
# filtered_data_mlp = data_mlp[data_mlp['bitwidth'] == 8]
filtered_data_dt = data_dt
filtered_data_svm = data_svm
filtered_data_mlp = data_mlp



dt4_bit = filtered_data_dt[filtered_data_dt['bitwidth'] == 4]['Power'].mean()
dt6_bit = filtered_data_dt[filtered_data_dt['bitwidth'] == 6]['Power'].mean()
dt8_bit = filtered_data_dt[filtered_data_dt['bitwidth'] == 8]['Power'].mean()
dt10_bit = filtered_data_dt[filtered_data_dt['bitwidth'] == 10]['Power'].mean()

dt4_bit_acc = filtered_data_dt[filtered_data_dt['bitwidth'] == 4]['Accuracy'].mean()
dt6_bit_acc = filtered_data_dt[filtered_data_dt['bitwidth'] == 6]['Accuracy'].mean()
dt8_bit_acc = filtered_data_dt[filtered_data_dt['bitwidth'] == 8]['Accuracy'].mean()
dt10_bit_acc = filtered_data_dt[filtered_data_dt['bitwidth'] == 10]['Accuracy'].mean()
# print(dt4_bit, dt6_bit, dt8_bit, dt10_bit)
# print(dt4_bit_acc, dt6_bit_acc, dt8_bit_acc, dt10_bit_acc)
# Step 3: Create a folder 'fig' if it doesn't exist
output_folder = 'fig'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 4: Define a marker style dictionary for each FeatureSelection
marker_styles = {
    'disr': '1',          # Dot marker for 'disr'
    'fisher': '1',  # Renamed from 'fisher_score' to 'fs' with star marker
    'jmi': '1'            # Plus marker for 'jmi'
}

# Define pastel colors from seaborn for the specific 'Number' values
vivid_colors = sns.color_palette("pastel", 10)
number_colors = {
    4: vivid_colors[6],  # Bright red
    6: vivid_colors[1],  # Bright blue
    8: vivid_colors[0],  # Bright green
    10: vivid_colors[3],  # Bright purple
}

# Define the color map function to assign pastel colors based on 'Number'
def color_map(number):
    return number_colors.get(number, 'lightgray')  # Lightgray for other numbers

# Step 5: Set up subplots (2 rows, 2 columns), with total width of 3.37 inches per subplot
# fig = plt.figure(figsize=(3.37, 2.5))  # Total width: 3.37in per subplot, 2 rows
fig = plt.figure(figsize=(3.37, 1.4))  # Total width: 3.37in per subplot, 2 rows
gs = GridSpec(2, 2, figure=fig)  # First row: 1 unit high, second row: 2 units high
ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
ax1 = fig.add_subplot(gs[0, 1])  # First row, second column
ax2 = fig.add_subplot(gs[1, :])
# Step 6: Reduce space between subplots to maximize figure area
plt.subplots_adjust(left=0.1, right=0.96, top=0.89, bottom=0.1, wspace=0.25, hspace=0.4)

# Step 7: Plot for modelDT in the first subplot (ax0)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_dt[filtered_data_dt['FeatureSelection'] == feature_selection]
    colors = [color_map(n) for n in subset['bitwidth']]  # Apply color map to each Number
    ax0.scatter(
        subset['Power'], 
        subset['Accuracy'], 
        c=colors,                # Assign colors based on the Number column
        marker=marker,           # Assign different markers
        s=50,                    # Marker size (adjusted for visibility)
        alpha=0.8,               # Transparency for visibility
        label=feature_selection  # Add label for shared legend
    )

# Calculate ticks for x and y axes for DT plot
x_min_dt, x_max_dt = filtered_data_dt['Power'].min(), filtered_data_dt['Power'].max()
y_min_dt, y_max_dt = filtered_data_dt['Accuracy'].min(), filtered_data_dt['Accuracy'].max()
ax0.set_xticks(np.round(np.linspace(x_min_dt, x_max_dt, 3), 4))
ax0.set_yticks(np.round(np.linspace(y_min_dt, y_max_dt, 5)))
ax0.tick_params(axis='both', labelsize=7, pad=0.2)
ax0.text(0.7, 0.1, 'a) DT', ha='left', va='center', transform=ax0.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_dt['Power'], filtered_data_dt['Accuracy'], maxX=False, maxY=True)
ax0.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points
# print(pareto_x[-1], pareto_y[-1])
print()

q_list=[]
for i in range(len(pareto_x)):
    # print(pareto_x[i], pareto_y[i])
    # print(filtered_data_dt[(filtered_data_dt['Area'] == pareto_x[i]) & (filtered_data_dt['Accuracy'] == pareto_y[i])]['FeatureSelection'].values[0])
    print(filtered_data_dt[(filtered_data_dt['Power'] == pareto_x[i]) & (filtered_data_dt['Accuracy'] == pareto_y[i])]['bitwidth'].values[0])
    q_list.append(filtered_data_dt[(filtered_data_dt['Power'] == pareto_x[i]) & (filtered_data_dt['Accuracy'] == pareto_y[i])]['bitwidth'].values[0])
    # print()
# print

eight_bit = q_list.count(6)
total_count = len(q_list)
eight_bit_percentage = (eight_bit / total_count) * 100
print(f'8-bit Percentage: {eight_bit_percentage:.2f}%')
print('DT')
print()
# Step 8: Plot for modelSVM in the second subplot (ax1)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_svm[filtered_data_svm['FeatureSelection'] == feature_selection]
    colors = [color_map(n) for n in subset['bitwidth']]  # Apply color map to each Number
    ax1.scatter(
        subset['Power'], 
        subset['Accuracy'], 
        c=colors,                # Assign colors based on the Number column
        marker=marker,           # Assign different markers
        s=50,                    # Marker size (adjusted for visibility)
        alpha=0.8,               # Transparency for visibility
        label=feature_selection  # Add label for shared legend
    )

# Calculate ticks for x and y axes for SVM plot
x_min_svm, x_max_svm = filtered_data_svm['Power'].min(), filtered_data_svm['Power'].max()
y_min_svm, y_max_svm = filtered_data_svm['Accuracy'].min(), filtered_data_svm['Accuracy'].max()
ax1.set_xticks(np.round(np.linspace(x_min_svm, x_max_svm, 3), 3))
ax1.set_yticks(np.round(np.linspace(y_min_svm, y_max_svm, 5)))
ax1.tick_params(axis='both', labelsize=7, pad=0.2)
ax1.text(0.7, 0.1, 'b) SVM', ha='left', va='center', transform=ax1.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_svm['Power'], filtered_data_svm['Accuracy'], maxX=False, maxY=True)
ax1.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points
# print(pareto_x[-1], pareto_y[-1])

fs_list=[]
# q_list=[]
for i in range(len(pareto_x)):
    # print(pareto_x[i], pareto_y[i])
    # print(filtered_data_svm[(filtered_data_svm['Area'] == pareto_x[i]) & (filtered_data_svm['Accuracy'] == pareto_y[i])]['FeatureSelection'].values[0])
    fs_list.append(filtered_data_svm[(filtered_data_svm['Power'] == pareto_x[i]) & (filtered_data_svm['Accuracy'] == pareto_y[i])]['FeatureSelection'].values[0])
    print(filtered_data_svm[(filtered_data_svm['Power'] == pareto_x[i]) & (filtered_data_svm['Accuracy'] == pareto_y[i])]['bitwidth'].values[0])
    q_list.append(filtered_data_svm[(filtered_data_svm['Power'] == pareto_x[i]) & (filtered_data_svm['Accuracy'] == pareto_y[i])]['bitwidth'].values[0])
    # print()
# print
disr_count = fs_list.count('disr')
total_count = len(fs_list)
disr_percentage = (disr_count / total_count) * 100
# print(f'DISR Percentage: {disr_percentage:.2f}%')
eight_bit = q_list.count(6)
total_count = len(q_list)
eight_bit_percentage = (eight_bit / total_count) * 100
print(f'8-bit Percentage: {eight_bit_percentage:.2f}%')
print('SVM')
print()



# Step 9: Plot for modelMLP in the third subplot (ax2)
for feature_selection, marker in marker_styles.items():
    subset = filtered_data_mlp[filtered_data_mlp['FeatureSelection'] == feature_selection]
    colors = [color_map(n) for n in subset['bitwidth']]  # Apply color map to each Number
    ax2.scatter(
        subset['Power'], 
        subset['Accuracy'], 
        c=colors,                # Assign colors based on the Number column
        marker=marker,           # Assign different markers
        s=50,                    # Marker size (adjusted for visibility)
        alpha=0.8,               # Transparency for visibility
        label=feature_selection  # Add label for shared legend
    )

# Calculate ticks for x and y axes for MLP plot
x_min_mlp, x_max_mlp = filtered_data_mlp['Power'].min(), filtered_data_mlp['Power'].max()
y_min_mlp, y_max_mlp = filtered_data_mlp['Accuracy'].min(), filtered_data_mlp['Accuracy'].max()
ax2.set_xticks(np.round(np.linspace(x_min_mlp, x_max_mlp, 3), 3))
ax2.set_yticks(np.round(np.linspace(y_min_mlp, y_max_mlp, 5)))
ax2.tick_params(axis='both', labelsize=7, pad=0.2)
ax2.text(0.87, 0.1, 'c) MLP', ha='left', va='center', transform=ax2.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_mlp['Power'], filtered_data_mlp['Accuracy'], maxX=False, maxY=True)
ax2.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points
print(pareto_x[-1], pareto_y[-1])
# print(pareto_x[-2], pareto_y[-2])
# print(pareto_x[-3], pareto_y[-3])
print(pareto_x[-1], pareto_y[-1])
print(pareto_x[-2], pareto_y[-2])
print()
# quit()
print()

fs_list=[]
# q_list=[]
sparsity_rate_list=[]
print(filtered_data_mlp)
for i in range(len(pareto_x)):
    # print(pareto_x[i], pareto_y[i])
    # print(filtered_data_mlp[(filtered_data_mlp['Area'] == pareto_x[i]) & (filtered_data_mlp['Accuracy'] == pareto_y[i])]['FeatureSelection'].values[0])
    fs_list.append(filtered_data_mlp[(filtered_data_mlp['Power'] == pareto_x[i]) & (filtered_data_mlp['Accuracy'] == pareto_y[i])]['FeatureSelection'].values[0])
    # print(filtered_data_mlp[(filtered_data_mlp['Area'] == pareto_x[i]) & (filtered_data_mlp['Accuracy'] == pareto_y[i])]['bitwidth'].values[0])
    q_list.append(filtered_data_mlp[(filtered_data_mlp['Power'] == pareto_x[i]) & (filtered_data_mlp['Accuracy'] == pareto_y[i])]['bitwidth'].values[0])
    sparsity_rate_list.append(filtered_data_mlp[(filtered_data_mlp['Power'] == pareto_x[i]) & (filtered_data_mlp['Accuracy'] == pareto_y[i])]['sparsity_rate'].values[0])
    # print()
# print
disr_count = fs_list.count('disr')
total_count = len(fs_list)
disr_percentage = (disr_count / total_count) * 100
# print(f'DISR Percentage: {disr_percentage:.2f}%')
four_bit = q_list.count(4)
six_bit = q_list.count(6)
eight_bit = q_list.count(8)
ten_bit = q_list.count(10)
total_count = len(q_list)
four_bit_percentage = (four_bit / total_count) * 100
six_bit_percentage = (six_bit / total_count) * 100
eight_bit_percentage = (eight_bit / total_count) * 100
ten_bit_percentage = (ten_bit / total_count) * 100
print(f'4-bit Percentage: {four_bit_percentage:.2f}%')
print(f'6-bit Percentage: {six_bit_percentage:.2f}%')
print(f'8-bit Percentage: {eight_bit_percentage:.2f}%')
print(f'10-bit Percentage: {ten_bit_percentage:.2f}%')
print('MLP')
zero_2 = sparsity_rate_list.count(0.2)
zero_5 = sparsity_rate_list.count(0.5)
zero_9 = sparsity_rate_list.count(0.9)
total_count = len(sparsity_rate_list)
zero_2_percentage = (zero_2 / total_count) * 100
zero_5_percentage = (zero_5 / total_count) * 100
zero_9_percentage = (zero_9 / total_count) * 100
# print(sparsity_rate_list)
print(f'Sparsity 0.2: {zero_2_percentage:.2f}%')
print(f'Sparsity 0.5: {zero_5_percentage:.2f}%')
print(f'Sparsity 0.9: {zero_9_percentage:.2f}%')
print()
# print(q_list)
# Step 10: Remove individual y-axis labels and add a shared y-axis label for 'Accuracy'
fig.text(0.005, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=7)

# Step 11: Add a shared x-axis label for 'Power'
# fig.text(0.5, 0.01, 'Area (cm$^2$)', ha='center', fontsize=7)
# fig.text(0.5, 0.01, 'Power ($mW$)', ha='center', fontsize=7)

# Step 12: Create a combined legend with 3 rows and 6 columns
legend_handles = []

# Add legend entries for precisions
for number, color in number_colors.items():
    handle = plt.Line2D([0], [0], marker='_', color=color, markersize=5, linestyle='None', label=f'{number}-bits')
    legend_handles.append(handle)

# Add legend entries for feature selection algorithms
for feature_selection, marker in marker_styles.items():
    if feature_selection == 'fisher':
        handle = plt.Line2D([0], [0], marker=marker, color='black', markersize=7, linestyle='None', label=feature_selection.capitalize())
    else:
        handle = plt.Line2D([0], [0], marker=marker, color='black', markersize=7, linestyle='None', label=feature_selection.upper())
    legend_handles.append(handle)

legend_handles_reorder= [legend_handles[i] for i in [0, 4, 1, 5, 2, 6, 3]]
# Create the combined legend with 2 rows and 4 columns
fig.legend(
    handles=legend_handles[0:4], 
    loc='upper center', 
    fontsize=7, 
    ncol=4, 
    frameon=False, 
    prop = { 'size': 7, 'family': TITLE_FONT, 'weight': TITLE_WEIGHT },
    # bbox_to_anchor=(0.5, 1.025),
    bbox_to_anchor=(0.5, 1.04),
    handletextpad=-0.5,  # Reducing space between the symbol and text
    columnspacing=-0.02  # Reducing space between columns
)

# fig.legend(
#     handles=legend_handles[4:], 
#     loc='upper center', 
#     fontsize=7, 
#     ncol=3, 
#     frameon=False, 
#     bbox_to_anchor=(0.5, 0.96),
#     handletextpad=-0.5,  # Reducing space between the symbol and text
#     columnspacing=-0.02  # Reducing space between columns
# )

# Step 13: Save the figure as a PDF in the 'fig' folder
# output_pdf_path = os.path.join(output_folder, 'lbrPower__AffectiveROAD.pdf')
output_pdf_path = os.path.join(output_folder, 'lbrPower.pdf')
# output_pdf_path = os.path.join(output_folder, 'lbrPower.jpeg')
# plt.savefig(output_pdf_path, format='jpeg')
plt.savefig(output_pdf_path, format='pdf')

# Show the plot for visual confirmation (optional)
# plt.show()

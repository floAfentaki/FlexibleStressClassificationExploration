import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
            if pair[1] >= p_front[-1][1]:  # Keep the point if it has a higher Accuracy
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:  # Keep the point if it has a lower Accuracy
                p_front.append(pair)
    
    # Extract the X and Y values into separate lists
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    
    return p_frontX, p_frontY

# Step 1: Load the Prune MLP CSV file
csv_file_prune_mlp = 'model_Prune_MLP.csv'  # Replace with the path to your actual model_Prune_MLP CSV file
data_prune_mlp = pd.read_csv(csv_file_prune_mlp)

# Step 2: Create a folder 'fig' if it doesn't exist
output_folder = 'fig'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 3: Define the marker style for each FeatureSelection
marker_styles = {
    'l2': '.',          # Dot marker for 'l2'
    'hessian': '*',     # Star marker for 'hessian'
    'wanda': '+'        # Plus marker for 'wanda'
}

# Step 4: Define pastel colors for the different sparsity rates using seaborn color palette
sparsity_rates = sorted(data_prune_mlp['sparsity_rate'].unique())  # Get unique sparsity rates
sparsity_colors = sns.color_palette("pastel", len(sparsity_rates))  # Use pastel colors for sparsity rates
sparsity_color_map = {rate: color for rate, color in zip(sparsity_rates, sparsity_colors)}

# Step 5: Create a single scatter plot
fig, ax = plt.subplots(figsize=(3.37, 1.5))  # Only one subplot
plt.subplots_adjust(left=0.09, right=0.96, top=0.80, bottom=0.17, wspace=0.25, hspace=0.25)

# Step 6: Plot for the pruning data (model_Prune_MLP)
legend_handles = []  # List to hold legend handles
for prune_method, marker in marker_styles.items():
    subset = data_prune_mlp[data_prune_mlp['PruneMethod'] == prune_method]
    
    # Iterate over each unique 'sparsity_rate' and assign colors
    for sparsity_rate in subset['sparsity_rate'].unique():
        subset_sparsity = subset[subset['sparsity_rate'] == sparsity_rate]
        color = sparsity_color_map[sparsity_rate]  # Assign pastel color based on sparsity_rate
        
        # Plot on the single subplot
        ax.scatter(
            subset_sparsity['Area'], 
            subset_sparsity['Accuracy'], 
            c=[color] * len(subset_sparsity),  # Use the pastel color for this sparsity rate
            marker=marker,                    # Different marker for each FeatureSelection
            s=50,                             # Marker size
            alpha=0.8,                        # Transparency for visibility
            label=f'{prune_method} {sparsity_rate}'  # Label includes FeatureSelection and sparsity_rate
        )

        # Create legend handles for each combination of prune_method and sparsity_rate
        handle = plt.Line2D(
            [0], [0], marker=marker, color=color, markersize=7, linestyle='None', 
            label=f'{prune_method} {sparsity_rate}'
        )
        legend_handles.append(handle)

# Step 6.1: Highlight Pareto points
pareto_x, pareto_y = pareto_frontier_2d(data_prune_mlp['Area'], data_prune_mlp['Accuracy'], maxX=False, maxY=True)
ax.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)  # Connect Pareto points

# Step 7: Set axis labels and ticks
ax.set_xlabel('Area (cm^2)', fontsize=7, labelpad=0.2)
ax.set_ylabel('Accuracy', fontsize=7, labelpad=0.2)
# ax.tick_params(axis='both', labelsize=7, pad=0.1)

# Calculate ticks for x and y axes
x_min, x_max = data_prune_mlp['Area'].min(), data_prune_mlp['Area'].max()
y_min, y_max = data_prune_mlp['Accuracy'].min(), data_prune_mlp['Accuracy'].max()
ax.set_xticks(np.round(np.linspace(x_min, x_max, 4), 3))
ax.set_yticks(np.round(np.linspace(y_min, y_max, 5)))
ax.tick_params(axis='both', labelsize=7, pad=0.1)
ax.text(0.87, 0.1, 'a) MLP', ha='left', va='center', transform=ax.transAxes, fontsize=7)

# Step 8: Create a legend
fig.legend(
    handles=legend_handles, 
    loc='upper center', 
    fontsize=7, 
    ncol=3, 
    frameon=False, 
    handleheight=0.0,  # Adjust the height of the legend handles
    labelspacing=0.0,  # Reducing space between the rows
    bbox_to_anchor=(0.5, 1.06)
)

# Step 9: Save the figure as 'pruning.pdf' in the 'fig' folder
output_pdf_path = os.path.join(output_folder, 'pruning.pdf')
plt.savefig(output_pdf_path, format='pdf')

print(f"Pruning plot saved as {output_pdf_path}")

# Optional: Show the plot for visual confirmation
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec


def pareto_frontier_2d(Xs, Ys, maxX=False, maxY=True):
    Xs = Xs.tolist()
    Ys = Ys.tolist()
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    p_front = [myList[0]]    
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


csv_file_dt = 'modelDT.csv'
csv_file_svm = 'modelSVM.csv'
csv_file_mlp = 'modelMLP.csv'

data_dt = pd.read_csv(csv_file_dt)
data_svm = pd.read_csv(csv_file_svm)
data_mlp = pd.read_csv(csv_file_mlp)

data_frontend = pd.read_csv("frontends_eval.csv")

data_dt['FeatureSelection'] = data_dt['FeatureSelection'].replace('fisher_score', 'fisher')
data_svm['FeatureSelection'] = data_svm['FeatureSelection'].replace('fisher_score', 'fisher')
data_mlp['FeatureSelection'] = data_mlp['FeatureSelection'].replace('fisher_score', 'fisher')

data_dt = pd.merge(
    data_dt,
    data_frontend[['fs_num', 'BITWIDTH', 'ADC_Area(um2)', 'FE_Area(um2)','FE_Power(mW)' ,'ADC_Power(mW)']],
    how='left',
    left_on=['Number', 'bitwidth'],
    right_on=['fs_num', 'BITWIDTH']
)

data_dt['ADC_Area(cm2)']=data_dt['ADC_Area(um2)']/1e8
data_dt['FE_Area(cm2)']=data_dt['FE_Area(um2)']/1e8
data_dt['Total_Area'] = data_dt['Area'] + data_dt['FE_Area(cm2)']+ data_dt['ADC_Area(cm2)']
data_dt['FE_ratio']=data_dt['FE_Area(cm2)']/data_dt['Total_Area']
print(data_dt['FE_ratio'].max())
print(data_dt['Total_Area'].describe())
print(data_dt['Accuracy'].describe())
bitwidth_2 = data_dt[data_dt['bitwidth'] == 2]
print(bitwidth_2['Total_Area'].describe())
print(bitwidth_2['Accuracy'].describe())
bitwidth_30_35_40 = data_dt[data_dt['Number'].isin([30, 35, 40])]
print(bitwidth_30_35_40['Total_Area'].describe())
print(bitwidth_30_35_40['Accuracy'].describe())

data_mlp = pd.merge(
    data_mlp,
    data_frontend[['fs_num', 'BITWIDTH', 'ADC_Area(um2)', 'FE_Area(um2)','FE_Power(mW)' ,'ADC_Power(mW)']],
    how='left',
    left_on=['Number', 'bitwidth'],
    right_on=['fs_num', 'BITWIDTH']
)

data_mlp['ADC_Area(cm2)']=data_mlp['ADC_Area(um2)']/1e8
data_mlp['FE_Area(cm2)']=data_mlp['FE_Area(um2)']/1e8
data_mlp['Total_Area'] = data_mlp['Area'] + data_mlp['FE_Area(cm2)']+ data_mlp['ADC_Area(cm2)']
data_mlp['FE_ratio']=data_mlp['FE_Area(cm2)']/data_mlp['Total_Area']
print(data_mlp['FE_ratio'].max())
print(data_mlp['Total_Area'].describe())
print(data_mlp['Accuracy'].describe())
bitwidth_2 = data_mlp[data_mlp['bitwidth'] == 2]
print(bitwidth_2['Total_Area'].describe())
print(bitwidth_2['Accuracy'].describe())
bitwidth_30_35_40 = data_mlp[data_mlp['Number'].isin([30, 35, 40])]
print(bitwidth_30_35_40['Total_Area'].describe())
print(bitwidth_30_35_40['Accuracy'].describe())

data_mlp['Area'] = data_mlp['Total_Area']

filtered_data_dt = data_dt
filtered_data_svm = data_svm
filtered_data_mlp = data_mlp
output_folder = 'fig'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

marker_styles = {
    2: 'x',
    4: '.',
    6: '+',
    8: '*',
}

pastel_colors = sns.color_palette("pastel", 6)
number_colors = {
    2: pastel_colors[0],
    4: pastel_colors[1],
    6: pastel_colors[2],
    8: pastel_colors[3],
}

def color_map(number):
    return number_colors.get(number, 'lightgray')

fig = plt.figure(figsize=(3.37, 2))
gs = GridSpec(2, 1, figure=fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
plt.subplots_adjust(left=0.09, right=0.9, top=0.9, bottom=0.12, wspace=0.25, hspace=0.25)

for feature_selection, marker in marker_styles.items():
    subset = filtered_data_dt[filtered_data_dt['bitwidth'] == feature_selection]
    colors = [color_map(n) for n in subset['bitwidth']]
    ax0.scatter(
        subset['Total_Area'], 
        subset['Accuracy'], 
        c=colors,
        marker=marker,
        s=15,
        alpha=0.8,
        label=feature_selection
    )

pareto={
    "0": {
        "accuracy": 100,
        "area_model": 9431.52,
        "power_model": 0.000539,
        "area_fs": 53256197.18,
        "power_fs": 1.673817,
        "area_adc": 38258127.2,
        "power_adc": 579.4138300000001,
    },
    "1": {
        "accuracy": 95.65,
        "area_model": 2630.60,
        "power_model": 0.000172,
        "area_fs": 14031653.329999996,
        "power_fs": 0.38886799999999994,
        "area_adc": 37900407.730000004,
        "power_adc": 574.29995,
    },
}
pareto_dt_areas = [52833602.93/ 1e8, 14297829.44/ 1e8]
pareto_dt_accuracies = [100, 96]
pareto_mlp_areas = [417965572.9/ 1e8, 148764773/ 1e8]
pareto_mlp_accuracies = [100, 96]

ax0.scatter(
    pareto_dt_areas, 
    pareto_dt_accuracies, 
    c='orange',
    marker='^',
    s=10,
    label='Ours Pareto',
    edgecolors='black',
)

ax0.text(
        pareto_dt_areas[0]+0.00002, 
        pareto_dt_accuracies[0]-5, 
        f"ours0", 
        fontsize=6, 
        ha='left', 
        va='center', 
        color='black'
    )
ax0.text(
        pareto_dt_areas[1]+0.00001, 
        pareto_dt_accuracies[1]-5, 
        f"ours1", 
        fontsize=6, 
        ha='left', 
        va='center', 
        color='black'
    )

x_min_dt, x_max_dt = filtered_data_dt['Total_Area'].min(), filtered_data_dt['Total_Area'].max()
y_min_dt, y_max_dt = filtered_data_dt['Accuracy'].min(), pareto_dt_accuracies[0]
ax0.set_xticks(np.round(np.linspace(x_min_dt, x_max_dt, 3), 4))
ax0.set_yticks(np.round(np.linspace(y_min_dt, y_max_dt, 5)))
ax0.tick_params(axis='both', labelsize=7, pad=0.2)
ax0.text(0.7, 0.8, 'a) DT', ha='left', va='center', transform=ax0.transAxes, fontsize=7)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_dt['Total_Area'], filtered_data_dt['Accuracy'], maxX=False, maxY=True)
ax0.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)
datafff=filtered_data_dt.where(filtered_data_dt['Total_Area']==max(pareto_x)).dropna()
print( datafff['Accuracy'].values[0], datafff['ADC_Area(cm2)'].values[0],    datafff['FE_Area(cm2)'].values[0], datafff['Area'].values[0],datafff['ADC_Power(mW)'].values[0], datafff['FE_Power(mW)'].values[0], datafff['Power'].values[0])

for feature_selection, marker in marker_styles.items():
    subset = filtered_data_mlp[filtered_data_mlp['bitwidth'] == feature_selection]
    colors = [color_map(n) for n in subset['bitwidth']]
    ax1.scatter(
        subset['Total_Area'], 
        subset['Accuracy'], 
        c=colors,
        marker=marker,
        s=15,
        alpha=0.8,
        label=feature_selection
    )

print(data_mlp['Total_Area'].describe())
print(data_mlp['Accuracy'].describe())
bitwidth_2 = data_mlp[data_mlp['bitwidth'] == 2]
bitwidth_30_35_40 = data_mlp[data_mlp['Number'].isin([30, 35, 40])]
print(bitwidth_30_35_40['Total_Area'].describe())
print(bitwidth_30_35_40['Accuracy'].describe())

x_min_mlp, x_max_mlp = filtered_data_mlp['Total_Area'].min(), filtered_data_mlp['Total_Area'].max()
y_min_mlp, y_max_mlp = filtered_data_mlp['Accuracy'].min(), filtered_data_mlp['Accuracy'].max()
print(y_max_mlp)
ax1.set_xticks(np.round(np.linspace(x_min_mlp, x_max_mlp, 3), 3))
ax1.set_yticks(np.round(np.linspace(y_min_mlp, 100, 5)))
ax1.tick_params(axis='both', labelsize=7, pad=0.2)
ax1.text(0.8, 0.8, 'b) MLP', ha='left', va='center', transform=ax1.transAxes, fontsize=7)

ax1.scatter(
    pareto_mlp_areas, 
    pareto_mlp_accuracies, 
    c='orange',
    marker='^',
    s=10,
    label='Ours Pareto',
    edgecolors='black',
)

ax1.text(
        pareto_mlp_areas[0]+0.0011, 
        pareto_mlp_accuracies[0]-4, 
        f"ours0", 
        fontsize=6, 
        ha='left', 
        va='center', 
        color='black'
    )
ax1.text(
        pareto_mlp_areas[1]+0.01,
        pareto_mlp_accuracies[1]-4, 
        f"ours1", 
        fontsize=6, 
        ha='left', 
        va='center', 
        color='black'
)

pareto_x, pareto_y = pareto_frontier_2d(filtered_data_mlp['Total_Area'], filtered_data_mlp['Accuracy'], maxX=False, maxY=True)
ax1.plot(pareto_x, pareto_y, linestyle='-', color='gray', lw=2, alpha=0.8)

fig.text(0.0008, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=7)
fig.text(0.5, 0.01, 'Area(cm^2)', ha='center', fontsize=7)

legend_handles = []
for number, color in number_colors.items():
        handle = plt.Line2D([0], [0], marker=marker_styles[number], color=color, markersize=7, linestyle='None', label=f'{number}-bits')
        legend_handles.append(handle)

fig.legend(
    handles=legend_handles, 
    loc='upper center', 
    fontsize=7, 
    ncol=4, 
    frameon=False, 
    bbox_to_anchor=(0.5, 1.025),
    handletextpad=-0.5,
    columnspacing=-0.02
)

output_pdf_path = os.path.join(output_folder, 'sota_quant.pdf')
plt.savefig(output_pdf_path, format='pdf')

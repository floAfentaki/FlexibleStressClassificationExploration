import pandas as pd
import numpy as np

# Load CSV files
csv1_path = '/home/flo/Documents/confs/DATE25/evaluation/Healthcare_PE/MLP/wesad/disr/float_models_8_synth/csv.csv'
csv2_path = '/home/flo/Documents/confs/DATE25/evaluation/Healthcare_PE/MLP/wesad/disr/pruning_ft_l2_8_synth/csv.csv'

df1 = pd.read_csv(csv1_path, delimiter=';')
df2 = pd.read_csv(csv2_path, delimiter=';')
df1=df1.dropna(axis=1)
df2=df2.dropna(axis=1)
print(df2)
column_to_proc=df2['Number']
df2['Number'] = df2['Number'].apply(lambda x: x.split('_')[0]).astype(int)
df2['Sparsity'] = column_to_proc.apply(lambda x: x.split('_')[3]).astype(int)
print(df2)


for i in [2, 5, 9]:
    dtemp= df2[df2['Sparsity']== i]
    # print(dtemp)
    area_drop_lst, original_area_lst, prubed_area_lst=[], [], []
    for y in [5, 10, 15, 20, 25, 30]:
        dtemp2=dtemp[dtemp['Number'] == y]
        original_area = df1[df1['Number'] == y]['Area'].values[0]
        original_area_lst.append(original_area)
        pruned_area = dtemp2['Area'].values[0]
        prubed_area_lst.append(pruned_area)
        area_drop = original_area - pruned_area
        area_drop_lst.append(area_drop)
        # print(f"Original area: {original_area}, Pruned area: {pruned_area}, Area drop: {area_drop}")
        # print(dtemp2)
    dsdsds = pd.DataFrame({
        'Sparsity': i,
        'area_drop_lst': area_drop_lst,
        'original_area_lst': original_area_lst, 
        'prubed_area_lst': prubed_area_lst
    })    

    dsdsds['area_reduction_percentage'] = (dsdsds['area_drop_lst'] / dsdsds['original_area_lst']) * 100
    print(dsdsds)
    area_reduction_percentage = (sum(area_drop_lst) / sum(original_area_lst)) * 100
    print(f"Total area reduction percentage for sparsity {i}: {np.round(area_reduction_percentage,1)}%")
    print(f"Percentage of area reduction  for sparsity {i}: {dsdsds['area_reduction_percentage'].mean()}%")

    # print(df2[df2['Sparsity'] == i]['Area'].sum())
# print(dsdsds)
# Find corresponding Area in df1 for different sparsity
df1_area_dict = df1.set_index('Number')['Area'].to_dict()

# Calculate area drop for different sparsity
filtered_df['area_drop'] = filtered_df['Number'].apply(lambda x: df1_area_dict.get(x, 0)) - filtered_df['Area_pruned']
# Merge dataframes on 'Number' column
merged_df = pd.merge(df1, df2, on='Number', suffixes=('_float', '_pruned'))
# print(merged_df)

# Calculate area drop for different sparsity
merged_df['area_drop'] = merged_df['Area_float'] - merged_df['Area_pruned']

# Filter rows where sparsity is different
print(merged_df)

filtered_df = merged_df[merged_df['sparsity_float'] != merged_df['sparsity_pruned']]

# Display the results
print(filtered_df[['Number', 'sparsity_float', 'sparsity_pruned', 'area_drop']])
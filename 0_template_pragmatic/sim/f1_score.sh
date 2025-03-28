#!/bin/bash

# Read predictions and true labels from files
y_pred_tensor=$(cat output.txt)
y_true_tensor=$(cat sim.Ytest)

# Convert space-separated strings to arrays
y_pred_array=($y_pred_tensor)
y_true_array=($y_true_tensor)

# Initialize counters
TP=0
FP=0
FN=0

# Calculate TP, FP, FN
for i in "${!y_pred_array[@]}"; do
    if [[ ${y_pred_array[$i]} -eq 1 && ${y_true_array[$i]} -eq 1 ]]; then
        TP=$((TP + 1))
    elif [[ ${y_pred_array[$i]} -eq 1 && ${y_true_array[$i]} -eq 0 ]]; then
        FP=$((FP + 1))
    elif [[ ${y_pred_array[$i]} -eq 0 && ${y_true_array[$i]} -eq 1 ]]; then
        FN=$((FN + 1))
    fi
done

# Calculate precision, recall, and F1 score
if [[ $((TP + FP)) -gt 0 ]]; then
    precision=$(echo "scale=4; $TP / ($TP + $FP)" | bc)
else
    precision=0
fi

if [[ $((TP + FN)) -gt 0 ]]; then
    recall=$(echo "scale=4; $TP / ($TP + $FN)" | bc)
else
    recall=0
fi

if [[ $(echo "$precision + $recall > 0" | bc) -eq 1 ]]; then
    f1=$(echo "scale=4; 2 * ($precision * $recall) / ($precision + $recall)" | bc)
else
    f1=0
fi

# Output the F1 score
echo "F1 Score: $f1"

cd reports
area=$(cat *ns.area.rpt | grep 'Total cell area:' | awk '{printf "%.5f", ($NF/10^8)}')
power=$(cat *ns.power.ptpx.rpt | grep 'Total Power' | awk '{printf "%.2f", ($4*10^3)}')
cd ../sim
acc=$(paste output.txt sim.Ytest | awk 'BEGIN{score=0}{if ($1==$2) score++;}END{print (score/NR)*10^2}')
cd ../reports
echo -e "Accuracy: $acc\tArea: $area cm^2\tPower: $power mW">reports.txt

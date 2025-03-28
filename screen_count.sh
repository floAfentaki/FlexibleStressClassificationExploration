#!/bin/bash

active_screens=$(screen -list | grep -o '[0-9]\{4,8\}\.')
cnt=0
for screen in $active_screens; do  
    cnt=$((cnt + 1))  
done
echo "Number of active screens: $cnt"
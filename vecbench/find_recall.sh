#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




if [[ $# -ne 7 ]]; then
	echo "Found Arguments: $#"
	echo "USAGE: bash find_recall.sh <config_file> <base_search> <recall_type> <max_runs> <range> <epsilon> <reordering>"
	echo "1. <config_file> = relative path of the benchmark config file eg. 'config/benchmark/simple_ann_tree_ah_cosine.yaml'
		2. <base_search> = The first value of num_leaves_to_search
		3. <recall_type> = They type of recall that should be 95 (One value out of 'Matching neighbor IDs only', 'Comparing neighbor distances only', and 'Matching neighbors IDs and considering distance ties')
		4. <max_runs> = The number of maximum runs to be taken 
		5. <range> = range of num_leaves_to_search (The script will perform a binary search to find the optimal num_leaves_to_search that gives us a 96 recall between max(0, base_search-range) and base_search + range)
		6. <epsilon> = The allowed additional recall value divided by 100 (if the epsilon is 0.001 then the script will perform a binary search until the obtained recall is between 95 and 95.1)
		7. <reordering> = whether this benchmark has pre_reordering_num_neighbors. Enter 'reorder' if yes, 'noreorder' otherwise"
	exit 1
fi

config_file=$1
base_search=$2
recall_type=$3
line=$(cat $config_file | grep "num_leaves_to_search")
newline="${line%:*}: "${base_search}
sed -i "s/${line}/${newline}/g" $config_file
run_num=0
recall=0
max_runs=$4
range=$5
m=$base_search
l=0
if [[ $m -gt $range ]]; then 
	l=$(( $base_search - $range ))
fi
h=$(( $base_search + $range))
epsilon=$6
reorder=$7
while [[ $run_num -lt $max_runs && ( $(echo "$recall < 0.95" | bc) -eq 1 || $(echo "$recall > 0.95 + $epsilon" | bc) -eq 1 ) ]]; do
	outfile='output_'${run_num}'.txt'
	line=$(cat $config_file | grep "num_leaves_to_search")
	newline="${line%:*}: "${m}
	sed -i "s/${line}/${newline}/g" $config_file
	if [[ $7 = "reorder" ]];then
		echo "Reordering is True"
		num_neigh=$(python3 find_corresp_neighbors.py $m)
		echo "Setting Reordeing Neighbors as $num_neigh"
		line=$(cat $config_file | grep "pre_reordering_num_neighbors")
        	newline="${line%:*}: "${num_neigh}
        	sed -i "s/${line}/${newline}/g" $config_file
	fi
	echo "Running with ${m} Leaves to search"
	./run.sh > $outfile
	recall_line=$(cat $outfile | grep "${3}")
	recall_string="${recall_line#*:}"
	echo "Current Recall is ${recall_string}"
	recall=$( echo "${recall_string}" | bc )
	if [[ $( echo "$recall < 0.95" | bc ) -eq 1 ]]; then
		l=$(( m + 1 ))
	fi
	if [[ $( echo "$recall > 0.95 + $epsilon" | bc ) -eq 1 ]]; then
		h=$(( m - 1 ))
	fi
	if [[ $l -gt $h ]]; then
		echo "Breaking because l is $l and h is $h"
		break 
	fi
	m=$(( (l + h) / 2 ))
	run_num=$(( run_num + 1 ))
done


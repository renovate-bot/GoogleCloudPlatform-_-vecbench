# find_recall.sh

This script calls the run.sh script until 95% recall is obtained for scann index type. 
To run this script 

`bash find_recall.sh <config_file> <base_search> <recall_type> <max_runs> <range> <epsilon> <reordering>`

Where:
1. <config_file> = relative path of the benchmark config file eg. 'config/benchmark/simple_ann_tree_ah_cosine.yaml'
2. <base_search> = The first value of num_leaves_to_search
3. <recall_type> = They type of recall that should be 95 (One value out of 'Matching neighbor IDs only', 'Comparing neighbor distances only', and 'Matching neighbors IDs and considering distance ties')
4. <max_runs> = The maximum number of runs to be taken 
5. \<range> = range of num_leaves_to_search (The script will perform a binary search to find the optimal num_leaves_to_search that gives us a 95 recall between max(0, base_search-range) and base_search + range)
6. \<epsilon> = The allowed additional recall value divided by 100 (if the epsilon is 0.001 then the script will perform a binary search until the obtained recall is between 95 and 95.1)
7. \<reordering> = whether this benchmark has pre_reordering_num_neighbors. Enter 'reorder' if yes, 'noreorder' otherwise"


Packages required:
`sudo apt-get install bc`

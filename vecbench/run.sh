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

# DB_CONFIG='config/db/alloydb.yaml'
# DB_CONFIG='config/db/csqlpg.yaml'
# DB_CONFIG='config/db/alloydb-omni.yaml'
DB_CONFIG='config/db/alloydb-omni2.yaml'
# DB_CONFIG='config/db/pinecone-serverless.yaml'
# DB_CONFIG='config/db/vertex-vector-search.yaml'
# DB_CONFIG='config/db/memorystore.yaml'

DATASET_CONFIG='config/dataset/glove_100_angular.yaml'
# DATASET_CONFIG='config/dataset/cohere_10M_train.yaml'
# DATASET_CONFIG='config/dataset/bigann_uint8_10M_train.yaml'
# DATASET_CONFIG='config/dataset/bigann_uint8_100M_train.yaml'
# DATASET_CONFIG='config/dataset/bigann_uint8_1B_train.yaml'
# DATASET_CONFIG='config/dataset/laion_100M_train.yaml'
# DATASET_CONFIG='config/dataset/openai_5M_train.yaml'

# BENCHMARK_CONFIG='config/benchmark/calibrate.yaml'

### IVFFLAT BENCHMARKS
BENCHMARK_CONFIG='config/benchmark/simple_ann_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_10M_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_100M_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_1B_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_cohere_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_laion_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_openai_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_cohere_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_cohere_cosine_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_cohere_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_cohere_cosine_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_cohere_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_cohere_cosine_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_bigann_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_bigann_l2_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_bigann_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_bigann_l2_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_bigann_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_bigann_l2_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_bigann_l2_4_4_1.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_cohere_cosine_4_4_1.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_l2_0.1.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_l2_50.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_l2_90.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_cosine_0.1.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_cosine_50.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_cosine_90.yaml'

### HNSW BENCHMARKS
# BENCHMARK_CONFIG='config/benchmark/simple_ann_cosine_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_10M_l2_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_cohere_cosine_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_openai_cosine_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_cohere_cosine_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_cohere_cosine_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_cohere_cosine_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_bigann_l2_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_bigann_l2_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_bigann_l2_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_bigann_l2_1_1_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_bigann_l2_10_1_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_bigann_l2_1_10_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_cohere_cosine_1_1_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_cohere_cosine_10_1_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_cohere_cosine_1_10_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_ann_cosine_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_l2_0.1_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_l2_50_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_l2_90_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_cosine_0.1_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_cosine_50_hnsw.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_cosine_90_hnsw.yaml'

### TREEAH BENCHMARKS
# BENCHMARK_CONFIG='config/benchmark/simple_ann_tree_ah_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_tree_ah_10M_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_tree_ah_100M_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_tree_ah_1B_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_cohere_tree_ah_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_openai_tree_ah_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_laion_tree_ah_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_bigann_tree_ah_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_bigann_tree_ah_l2_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_bigann_tree_ah_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_bigann_tree_ah_l2_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_bigann_tree_ah_l2.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_bigann_tree_ah_l2_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_cohere_tree_ah_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_cohere_tree_ah_cosine_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_cohere_tree_ah_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_cohere_tree_ah_cosine_multiple.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_cohere_tree_ah_cosine.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_cohere_tree_ah_cosine_multiple.yaml '
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_tree_ah_l2_0.1.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_tree_ah_l2_1.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_tree_ah_l2_10.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_tree_ah_l2_90.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_tree_ah_cosine_0.1.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_tree_ah_cosine_1.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_tree_ah_cosine_10.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_tree_ah_cosine_90.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_bigann_tree_ah_l2_4_4_1.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_cohere_tree_ah_cosine_4_4_1.yaml'


### TREEAH NEXT24 BENCHMARKS
# BENCHMARK_CONFIG='config/benchmark/simple_glove_tree_ah_cosine_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_tree_ah_10M_l2_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_bigann_tree_ah_100M_l2_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_cohere_tree_ah_cosine_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/simple_openai_tree_ah_cosine_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_tree_ah_l2_0.1_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_tree_ah_l2_1_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_tree_ah_l2_10_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_bigann_tree_ah_l2_90_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_tree_ah_cosine_0.1_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_tree_ah_cosine_1_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_tree_ah_cosine_10_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/filtered_cohere_tree_ah_cosine_90_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_bigann_tree_ah_l2_4_4_1_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/mixed_cohere_tree_ah_cosine_4_4_1_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_bigann_tree_ah_l2_multiple_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_bigann_tree_ah_l2_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_bigann_tree_ah_l2_multiple_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_bigann_tree_ah_l2_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_bigann_tree_ah_l2_multiple_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_bigann_tree_ah_l2_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_cohere_tree_ah_cosine_multiple_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/delete_cohere_tree_ah_cosine_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_cohere_tree_ah_cosine_multiple_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/insert_cohere_tree_ah_cosine_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_cohere_tree_ah_cosine_multiple_next24.yaml'
# BENCHMARK_CONFIG='config/benchmark/update_cohere_tree_ah_cosine_next24.yaml'

EXPERIMENT_CONFIG='config/experiments/basic.yaml'

REPORT_FILE="tree-ah-glove.txt"
REPORT_FOLDER="odyssey-reports"

LOADER="MPLoader"
# LOADER="RAYLoader"

python3 vecbench.py  \
  --loader $LOADER \
  --db_config  $DB_CONFIG\
  --dataset_config  $DATASET_CONFIG\
  --benchmark_config  $BENCHMARK_CONFIG \
  --report_file $REPORT_FILE  \
  --report_folder $REPORT_FOLDER  \
  --metrics "PANDAS_METRICS" 
# --report_only Yes


# python3 vecbench.py  \
#   --experiment $EXPERIMENT_CONFIG \
#   --metrics "PANDAS_METRICS" \
#   --report_folder $REPORT_FOLDER

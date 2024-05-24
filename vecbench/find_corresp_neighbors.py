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

import sys
import logging

neighbor_map={10: 100, 15: 140, 25: 160, 35: 190, 40: 200, 45: 220, 50: 240, 60: 250, 70: 300, 80: 400, 100: 500, 120: 600, 150: 800, 200: 900, 250: 1000}

if len(sys.argv) != 2:
    logging.error("Usage: python3 find_corresp_neighbors.py <num_leaves_to_search>")
    exit(1)
num_leaves = int(sys.argv[1])

if num_leaves in neighbor_map:
    print(neighbor_map[num_leaves])
else:
    lower = num_leaves 
    while lower not in neighbor_map:
        lower -= 1
        if lower < 10:
            print(100)
            exit(0)
    lower_val = neighbor_map[lower]
    upper = num_leaves
    while upper not in neighbor_map:
        if upper > 250:
            print(1000)
            exit(0)
        upper += 1
    upper_val=neighbor_map[upper]
    print(int(lower_val + (upper_val - lower_val) * (num_leaves - lower) / (upper - lower)))




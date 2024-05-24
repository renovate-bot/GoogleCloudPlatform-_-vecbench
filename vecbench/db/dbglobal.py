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


class DBGlobal():
  L2_DISTANCE=0
  COSINE_SIMILARITY=1
  MAX_INNER_PRODUCT=2

  def algo_to_pred(algo):
    if algo == "vector_l2_ops":
        return DBGlobal.L2_DISTANCE
    elif algo == "vector_cosine_ops":
        return DBGlobal.COSINE_SIMILARITY
    elif algo == "vector_ip_ops":
        return DBGlobal.MAX_INNER_PRODUCT

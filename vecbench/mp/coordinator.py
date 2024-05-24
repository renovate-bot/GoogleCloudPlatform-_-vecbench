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

from multiprocessing import Barrier
import ray


class Coordinator:
  def __init__(self, number_of_clients, loader):
    self.loader = loader
    
    if "MPLoader" in self.loader:
      self.barrier = MPBarrier(number_of_clients) 
    elif "RAYLoader" in self.loader:
      self.barrier = RAYBarrier.remote(number_of_clients) 

  def block_and_wait(self):
    if "MPLoader" in self.loader:
      self.barrier = self.barrier.block_and_wait()
    elif "RAYLoader" in self.loader:
      self.barrier = self.barrier.block_and_wait.remote() 

class MPBarrier:
  def __init__(self, number_of_clients):
    self.barrier = Barrier(number_of_clients) 

  def block_and_wait(self):
    self.barrier.wait()

@ray.remote(num_cpus=0)
class RAYBarrier:
  def __init__(self, number_of_clients):
    self.barrier = Barrier(number_of_clients) 

  def block_and_wait(self):
    self.barrier.wait()


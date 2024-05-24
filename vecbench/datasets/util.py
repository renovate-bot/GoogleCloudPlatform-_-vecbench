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

import struct
from tqdm import tqdm
import pandas as pd
import deepdish as dd
import numpy 
import logging

logging.getLogger().setLevel(logging.INFO)


def make_hdf5_file(filetype, inputfile, hdf5_filename, key):
    if filetype == "binary":
        with open(inputfile, "rb") as fvec:
            num_points = struct.unpack("i", fvec.read(4))[0]
            num_dimensions = struct.unpack("i", fvec.read(4))[0]
            print(f"num_points: {num_points} num_dimensions: {num_dimensions}")
            vectors = np.frombuffer(
                fvec.read(num_points * num_dimensions), dtype=np.uint8
            ).reshape((num_points, num_dimensions))
    else:
        if filetype == "json":
            df = pd.read_json(inputfile, lines=True)
        elif filetype == "parquet":
            df = pd.read_parquet(inputfile, engine="pyarrow")
        else:
            logging.error(
                "Unsupported File Type. Valid file types are binary, json and parquet"
            )

        df.columns = ["id", "embedding"]
        num_points = len(df["id"])
        num_dimensions = len(df["embedding"].iloc[0])
        print(f"num_points: {num_points} num_dimensions: {num_dimensions}")
        vectors = numpy.zeros((num_points, num_dimensions))
        iterator = 0
        for emb in tqdm(df["embedding"], total=num_points):
            vectors[iterator] = numpy.array(emb)
            iterator += 1

    print(vectors.shape)
    d = {key: vectors}
    dd.io.save(hdf5_filename, d)

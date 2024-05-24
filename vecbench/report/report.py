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

import logging
from tinyflux import TinyFlux
from tinyflux import TagQuery
import pandas as pd
import os
import time
import glob
import shutil
from report.template_engine import render
from google.cloud import storage
import metrics
import random

class ReportMetrics:
    def __init__(self, benchmark_config):
        # list all csv files only
        run_id = benchmark_config['config']['run_id']
        self.db_file = f"downloads/db_{run_id}.csv"
        csv_files = glob.glob(f"downloads/db_{run_id}_*.csv")
        csv_files.sort(key=os.path.getmtime)
        print(f"Merging {len(csv_files)} files.")
        if len(csv_files) > 0:
            # Remove old data
            if os.path.exists(self.db_file):
                os.remove(self.db_file)

            benchstart = time.time()
            with open(self.db_file, "wb") as outfile:
                for filename in csv_files:
                    with open(filename, "rb") as infile:
                        shutil.copyfileobj(infile, outfile, -1)
                    os.remove(filename)
            benchend = time.time()
            print(f"Merge took:{(benchend-benchstart)}")

    def pd_from_csv(self):
        df = pd.read_csv(self.db_file, header=None)
        return df

    def reformat(self, df):
        colnum = 0
        coldrop = []
        numcols = len(df.columns)
        # Fix timestamp collision
        df[0] = df[0].apply(lambda x: f'{x}.000000' if '.' not in x else x)
        df.rename({0: "timestamp"}, axis=1, inplace=True)
        df.rename({1: "benchtype"}, axis=1, inplace=True)
        df.rename({numcols - 2: "fields"}, axis=1, inplace=True)
        df.rename({numcols - 1: "values"}, axis=1, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        row1 = df.values[:1][0]
        for colname in row1:
            if "_tag" in str(colname):
                newname = "_".join(colname.split("_")[2:])
                coldrop.append(colnum)
                df.rename({df.columns[colnum + 1]: newname}, axis=1, inplace=True)
            colnum = colnum + 1
        df = df.drop(df.columns[coldrop], axis=1)
        df.loc[:, "fields"] = df.fields.apply(lambda x: x[7:])
        return df


def generate_report(db_config, dataset_config, benchmark_config, report_file, report_folder=None):
    if metrics.PANDAS_METRICS in benchmark_config['config']['metrics']:
        run_id = benchmark_config['config']['run_id']
        csv_files = glob.glob(f"downloads/db_{run_id}_*.csv")
        if len(csv_files) == 0:
            logging.info("No Metrics DB")
            return
        reportmetrics = ReportMetrics(benchmark_config)
        df = reportmetrics.pd_from_csv()
        df = reportmetrics.reformat(df)

        if report_file is None:
            number_of_workers = benchmark_config['config']['number_of_workers']
            report_file=f"{db_config['type']}-{dataset_config['type']}-{benchmark_config['class']}-{number_of_workers}-{run_id}.txt"

        render(df, benchmark_config, report_file)
        if report_folder is not None:
            storage_client = storage.Client()
            bucket = storage_client.bucket(report_folder)
            blob = bucket.blob(report_file)
            blob.upload_from_filename(report_file)

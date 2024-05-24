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

from db.spanner.db import Spanner
import configparser
import datetime
import os
import uuid

import pytest

from sqlalchemy import (
    text,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    ForeignKey,
)

@pytest.fixture(scope="module")
def db():
    project = os.getenv(
        "GOOGLE_CLOUD_PROJECT",
        os.getenv("PROJECT_ID", "span-cloud-testing"),
    )
    config = {}
    config['project_id'] = project
    config['instance-id'] = 'test-instance-ziyue'
    config['database_id'] = 'vec-search-db'
    config['run_id'] = 1234
    return Spanner(config)

@pytest.fixture(scope="module")
def table_name():
    return "TestTable"

@pytest.fixture
def table(db, table_name):
    table = db.CreateTable(table_name, False, 0)
    yield table
    with db.engine.begin() as conn:
        conn.execute(table.delete())
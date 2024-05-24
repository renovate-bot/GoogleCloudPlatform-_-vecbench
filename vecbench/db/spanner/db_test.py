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
from sqlalchemy import (
    inspect,
    MetaData
)
import pytest
from db.dbglobal import DBGlobal

def test_create_table(capsys, db, table, table_name):
    assert inspect(db.engine).has_table(table_name) is True

def test_load_table(db, table, table_name):
    table = db.load_table(table_name)
    assert table is not None

def test_get_by_id(db, table, table_name):
    db.populate(table_name, [[1, [0.1, 0.2, 0.3]]], 1)
    db.load_table(table_name)
    assert db.get_by_id(1) == [(1, [0.1, 0.2, 0.3])]
    assert not db.get_by_id(2)

def test_populate_table(db, table, table_name):
    db.populate(table_name, [[1, [0.1, 0.2, 0.3]]], 1)
    rows = get_all_rows(db, table)
    print("Fetched rows: ", rows)
    assert len(rows) == 1
    assert rows[0] == (1, [0.1, 0.2, 0.3])

def test_populate_table_2(db, table, table_name):
    db.populate(table_name, [[0.1, 0.2, 0.3], [9, 0, 5.8], [100.3, 1, 100.2]], 1)
    rows = get_all_rows(db, table)
    print("Fetched rows: ", rows)
    assert len(rows) == 3
    assert rows == [(1, [0.1, 0.2, 0.3]), (2, [9, 0, 5.8]), (3, [100.3, 1, 100.2])]

def test_annsearch(db, table, table_name):
    db.populate(table_name, [[1, [9, 0, 5.8]]], 1)
    db.populate(table_name, [[2, [0.1, 0.2, 0.3]]], 1)
    db.populate(table_name, [[3, [100.3, 1, 100.2]]], 1)
    db.populate(table_name, [[4, [3, 10, 4]]], 1)
    rows = get_all_rows(db, table)
    rows = db.annsearch([2, 3, 4], 3, DBGlobal.COSINE_SIMILARITY).fetchall()
    print("Cosine distanced rows: ", rows)
    assert len(rows) == 3
    assert rows == [(2,), (4,), (3,)]
    rows = db.annsearch([2, 3, 4], 3, DBGlobal.L2_DISTANCE).fetchall()
    print("L2 distanced rows: ", rows)
    assert len(rows) == 3
    assert rows == [(2,), (4,), (1,)]

def test_annsearch_filtered(db, table, table_name):
    db.populate(table_name, [[1, [9, 0, 5.8]]], 1)
    db.populate(table_name, [[2, [0.1, 0.2, 0.3]]], 1)
    db.populate(table_name, [[3, [100.3, 1, 100.2]]], 1)
    db.populate(table_name, [[4, [3, 10, 4]]], 1)
    rows = get_all_rows(db, table)
    rows = db.annfilteredsearch(4, [2, 3, 4], 2, DBGlobal.COSINE_SIMILARITY).fetchall()
    print("Cosine distanced rows: ", rows)
    assert len(rows) == 2
    assert rows == [(2,), (3,)]
    rows = db.annfilteredsearch(4, [2, 3, 4], 2, DBGlobal.L2_DISTANCE).fetchall()
    print("L2 distanced rows: ", rows)
    assert len(rows) == 2
    assert rows == [(2,), (1,)]


def test_annsearch_filtered_mod(db, table, table_name):
    db.populate(table_name, [[1, [9, 0, 5.8]]], 1)
    db.populate(table_name, [[2, [0.1, 0.2, 0.3]]], 1)
    db.populate(table_name, [[3, [100.3, 1, 100.2]]], 1)
    db.populate(table_name, [[4, [3, 10, 4]]], 1)
    rows = get_all_rows(db, table)
    rows = db.annfilteredmodsearch(2, 0, [2, 3, 4], 2, DBGlobal.COSINE_SIMILARITY).fetchall()
    print("Cosine distanced rows: ", rows)
    assert len(rows) == 2
    assert rows == [(2,), (4,)]
    rows = db.annfilteredmodsearch(2, 0, [2, 3, 4], 2, DBGlobal.L2_DISTANCE).fetchall()
    print("L2 distanced rows: ", rows)
    assert len(rows) == 2
    assert rows == [(2,), (4,)]

def test_annsearch_range_filtered(db, table, table_name):
    db.populate(table_name, [[1, [9, 0, 5.8]]], 1)
    db.populate(table_name, [[2, [0.1, 0.2, 0.3]]], 1)
    db.populate(table_name, [[3, [100.3, 1, 100.2]]], 1)
    db.populate(table_name, [[4, [3, 10, 4]]], 1)
    rows = get_all_rows(db, table)
    rows = db.annfilteredrangesearch(1, 4, [2, 3, 4], 1, DBGlobal.COSINE_SIMILARITY).fetchall()
    print("Cosine distanced rows: ", rows)
    assert rows == [(2,)]
    rows = db.annfilteredrangesearch(1, 4, [2, 3, 4], 1, DBGlobal.L2_DISTANCE).fetchall()
    print("L2 distanced rows: ", rows)
    assert rows == [(2,)]

def test_annsearch_ops(db, table, table_name):
    assert db.anndatasetsize(table_name) == 0
    db.anninsert([9, 0, 5], table_name)
    db.anninsert([0.1, 0.2, 0.3], table_name)
    rows = get_all_rows(db, table)
    assert db.anndatasetsize(table_name) == 2
    assert rows == [(1, [9, 0, 5]), (2, [0.1, 0.2, 0.3])]
    db.annupdate(2, [2, 3, 4], table_name)
    rows = get_all_rows(db, table)
    assert db.anndatasetsize(table_name) == 2
    assert rows == [(1, [9, 0, 5]), (2, [2, 3, 4])]
    db.anndelete(1, table_name)
    rows = get_all_rows(db, table)
    assert db.anndatasetsize(table_name) == 1
    assert rows == [(2, [2, 3, 4])]
    db.anninsert([0.1, 0.2, 0.3], table_name)
    rows = get_all_rows(db, table)
    assert db.anndatasetsize(table_name) == 2
    assert rows == [(2, [2, 3, 4]), (3, [0.1, 0.2, 0.3])]


def get_all_rows(db, table):
    with db.engine.begin() as connection:
        rows = connection.execute(table.select()).fetchall()
    return rows
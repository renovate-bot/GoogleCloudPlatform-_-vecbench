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

import dataclasses
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import pymilvus

from db import dbglobal

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()

_ALGO_TO_METRIC_TYPE = {
    dbglobal.DBGlobal.L2_DISTANCE: "L2",
    dbglobal.DBGlobal.MAX_INNER_PRODUCT: "IP",
    dbglobal.DBGlobal.COSINE_SIMILARITY: "COSINE",
}


class Milvus:

    def __init__(self, config: Dict[Any, Any]):
        kwargs = {}
        if config.get("ip"):
            kwargs["host"] = config.get("ip")
        if config.get("port"):
            kwargs["port"] = config.get("port")
        if config.get("user"):
            kwargs["user"] = config.get("user")
        if config.get("password"):
            kwargs["password"] = config.get("password")
        if config.get("database"):
            kwargs["db_name"] = config.get("database")
        self.connection = pymilvus.connections.connect("default", **kwargs)
        self.shards_num: int = config.get("shards_num")
        self.type = "Milvus"
        self.vector_table: Optional[pymilvus.Collection] = None
        self.collections: Dict[str, pymilvus.Collection] = {}
        self.dataset_name: Optional[str] = None
        self.table_exists = False
        self.search_params = {}

    def load_table(self, table_name: str) -> pymilvus.Collection:
        collection = self._get_collection(table_name)
        self.vector_table = collection
        return collection

    def _get_collection(self, collection_name: str) -> pymilvus.Collection:
        if collection_name not in self.collections:
            self.collections[collection_name] = pymilvus.Collection(collection_name)
        return self.collections[collection_name]

    def CreateTable(
        self,
        table_name: str,
        db_recreate: bool,
        vector_dimension: int,
    ) -> pymilvus.Collection:
        if pymilvus.utility.has_collection(table_name):
            self.table_exists = True

        if db_recreate:
            pymilvus.utility.drop_collection(table_name)
            self.table_exists = False

        id = pymilvus.FieldSchema(
            name="id",
            dtype=pymilvus.DataType.INT64,
            is_primary=True,
        )
        embeddings = pymilvus.FieldSchema(
            name="embeddings",
            dtype=pymilvus.DataType.FLOAT_VECTOR,
            dim=vector_dimension,
        )
        schema = pymilvus.CollectionSchema(
            fields=[id, embeddings],
        )
        collection = pymilvus.Collection(
            name=table_name,
            schema=schema,
            using="default",
            shards_num=self.shards_num,
        )
        self.vector_table = collection
        return collection

    def populate(
        self,
        table_name: str,
        data: Sequence[Sequence[Any]],
        start: int,
    ):
        pid = os.getpid()
        self.dataset_name = table_name
        if self.table_exists:
            return
        logging.info("Populating table: %s with %d", table_name, len(data))
        if len(data[0]) == 2:
            self._populate_with_id(table_name, data)
        else:
            self._populate_without_id(table_name, data, start)

    def _populate_with_id(
        self,
        table_name: str,
        data: Sequence[Sequence[Any]],
    ):
        ids = []
        embeddings = []
        for id, embedding in data:
            ids.append(id)
            embeddings.append(embedding)
        self._get_collection(table_name).insert([ids, embeddings])

    def _populate_without_id(
        self,
        table_name: str,
        data: Sequence[Sequence[Any]],
        start: int,
    ):
        ids = list(range(start, start + len(data)))
        self._get_collection(table_name).insert([ids, data])

    def index_embeddings(
        self,
        benchmark_config: Dict[str, Any],
        vector_table: Any,
    ) -> None:
        index_recreate = benchmark_config["config"]["index_recreate"]
        index_type = benchmark_config["config"]["index_type"]
        index_config = benchmark_config["config"]["index_config"]
        algo = benchmark_config["config"]["algo"]

        if not index_recreate:
            return

        logging.info(
            "Indexing table type: %s with algo: %s config: %s",
            index_type,
            algo,
            index_config,
        )

        # Drop existing index if we have one.
        vector_table.drop_index()

        milvus_index_type, milvus_index_params = self._get_milvus_index_config(
            index_type, index_config
        )

        vector_table.create_index(
            field_name="embeddings",
            index_params={
                "metric_type": _ALGO_TO_METRIC_TYPE[
                    dbglobal.DBGlobal.algo_to_pred(algo)
                ],
                "index_type": milvus_index_type,
                "params": milvus_index_params,
            },
        )
        pymilvus.wait_for_index_building_complete(vector_table.name)
        # After index creation we need to load the collection into memory before
        # querying it.
        vector_table.load()

    def _get_milvus_index_config(
        self, index_type: str, index_config: Union[str, Dict]
    ) -> Tuple[str, str]:
        if not isinstance(index_config, Dict):
            raise ValueError(
                f"Benchmark index_config is not a dict, but a {type(str)}."
            )
        if index_type == "ivfflat":
            milvus_index_type = "IVF_FLAT"
            milvus_index_config = {
                "nlist": self._get_index_config_key(index_config, "lists")
            }
        elif index_type == "ivf":
            milvus_index_type = "IVF_SQ8"
            quantizer = self._get_index_config_key(index_config, "quantizer")
            if quantizer.lower() != "sq8":
                raise ValueError("Expected 'quantizer' to be 'SQ8'")
            milvus_index_config = {
                "nlist": self._get_index_config_key(index_config, "lists")
            }
        elif index_type == "hnsw":
            milvus_index_type = "HNSW"
            milvus_index_config = {
                "M": self._get_index_config_key(index_config, "m"),
                "efConstruction": self._get_index_config_key(
                    index_config, "ef_construction"
                ),
            }
        else:
            raise ValueError(f"Unsupported index type {index_type} on Milvus.")
        return milvus_index_type, milvus_index_config

    def _get_index_config_key(self, index_config: Dict, key: str) -> Any:
        value = index_config.get(key)
        if not value:
            raise ValueError(
                "Missing required property 'lists' in benchmark index_config."
            )
        return value

    def configure_search_session(
        self, benchmark_config: dict
    ) -> None:
        index_type = benchmark_config['index_type']
        probes = benchmark_config['probes']
        self.search_params = {}
        if index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            self.search_params["params"] = {
                "nprobe": probes,
            }
        elif index_type == "HNSW":
            self.search_params["params"] = {
                "ef": probes,
            }

    def annsearch(self, embedding: Any, limit: int, algo: int):
        search_params = self.search_params.copy()
        search_params["metric_type"] = _ALGO_TO_METRIC_TYPE[algo]
        return self.vector_table.search(
            data=[embedding],
            anns_field="embeddings",
            param=search_params,
            limit=limit,
            expr=None,
            output_fields=["id"],
            consistency_level="Strong",
        )

    def anninsert(self, embedding: Any, table_name: str, insert_id: Any = None):
        pass

    def annupdate(self, id: int, embedding: Any, table_name: str) -> Any:
        return self._get_collection(table_name).upsert([[id], [embedding]])

    def anndelete(self, id: int, table_name: str) -> Any:
        result = self._get_collection(table_name).delete(f"id == {id}")
        return DeleteResponse(rowcount=result)

    def annfilteredsearch(self, id: int, embedding: Any, limit: int, algo: int):
        search_params = self.search_params.copy()
        search_params["metric_type"] = _ALGO_TO_METRIC_TYPE[algo]
        return self.vector_table.search(
            data=[embedding],
            anns_field="embeddings",
            param=search_params,
            limit=limit,
            expr=f"id < {id}",
            output_fields=["id"],
            consistency_level="Strong",
        )

    def anndatasetsize(self, table_name: str):
        return self._get_collection(table_name).num_entities

    def set_value(self, table_name):
        pass

    def get_by_id(self, id: int) -> List[Tuple[int, Sequence[float]]]:
        results = self.vector_table.query(
            expr=f"id == {id}",
            offset=0,
            limit=1,
            output_fields=["id", "embeddings"],
        )
        if results:
            return [(r["id"], r["embeddings"]) for r in results]
        return []

    def get_by_id_batch(
        self, ids: Sequence[int]
    ) -> List[List[Tuple[int, Sequence[float]]]]:
        vectors = []
        for id in ids:
            vectors.append(self.get_by_id(id))
        return vectors

    def returned_rows(self, response) -> List[Tuple[int]]:
        hits = response[0]  # This is a response for the ANNs of a single point.
        return [(id,) for id in hits.ids]


@dataclasses.dataclass
class DeleteResponse:
    rowcount: int

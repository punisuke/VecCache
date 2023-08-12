import os
import time
from typing import Tuple, Union

import faiss
import numpy as np
import openai
from dotenv import load_dotenv

from vec_cache.data_models import StoredText

load_dotenv()


class VecCache:
    def __init__(
        self,
        ttl: int,
        openai_api_key: str = "",
        embedding_model_name: str = "text-embedding-ada-002",
        vector_size: int = 1536,
        distance_thresh: float = 0.5,
    ):
        self.ttl = ttl
        self.embedding_model_name = embedding_model_name
        self.index = self._setup_db(vector_size)
        self.texts: list[StoredText] = []
        self.distance_thresh = distance_thresh
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        # TODO use ttl

    def _setup_db(self, size: int):
        return faiss.IndexFlatL2(size)

    def _add_vector(self, text: str, vector: list[float]) -> None:
        self.index.add(np.array([vector]).astype("float32"))
        self.texts += [StoredText(text=text)]

    def _generate_vector(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=self.embedding_model_name)[
            "data"
        ][0]["embedding"]

    def store(self, text: str) -> None:
        vector = self._generate_vector(text)
        self._add_vector(text, vector)

    def store_with_vector(self, text: str, vector: list[float]) -> None:
        self._add_vector(text, vector)

    def _vector_search(
        self, vector: list[float], return_with_distance: bool
    ) -> Union[str, Tuple[str, float]]:
        if not self.texts:
            return ("", np.float32(0.0)) if return_with_distance else ""

        distance, i = self.index.search(np.array([vector], dtype="float32"), 1)
        stored_text = self.texts[i[0][0]]

        if (time.time() - stored_text.timestamp) > self.ttl:
            return ("", np.float32(0.0)) if return_with_distance else ""

        return (
            (stored_text.text, distance[0][0])
            if return_with_distance
            else stored_text.text
        )

    def search(
        self, text: str, return_with_distance: bool = False
    ) -> Union[str, Tuple[str, float]]:
        vector = self._generate_vector(text)
        return self._vector_search(vector, return_with_distance)

    def search_with_vector(
        self, vector: list[float], return_with_distance: bool = False
    ) -> Union[str, Tuple[str, float]]:
        return self._vector_search(vector, return_with_distance)

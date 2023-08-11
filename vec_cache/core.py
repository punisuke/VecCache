import os
from typing import Tuple, Union

import faiss
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()


class VecCache:
    def __init__(
        self,
        ttl: int,
        openai_api_key: str = "",
        embedding_model_name: str = "text-embedding-ada-002",
        vector_size: int = 1536,
    ):
        self.ttl = ttl
        self.embedding_model_name = embedding_model_name
        self.index = self._setup_db(vector_size)
        self.texts: list[str] = []
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        # TODO use ttl

    def _setup_db(self, size: int):
        return faiss.IndexFlatL2(size)

    def _add_vector(self, text: str, vector: list[float]) -> None:
        self.index.add(np.array([vector]).astype("float32"))
        self.texts += [text]

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

    def search(
        self, text: str, return_with_distance: bool = False
    ) -> Union[str, Tuple[str, float]]:
        vector = self._generate_vector(text)
        distance, i = self.index.search(np.array([vector]).astype("float32"), 1)
        if return_with_distance:
            return self.texts[i[0][0]], distance
        else:
            return self.texts[i[0][0]]

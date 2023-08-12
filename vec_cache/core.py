import os
import time
from typing import List, Tuple, Union

import faiss
import numpy as np
import openai
from dotenv import load_dotenv

from vec_cache.data_models import StoredText

load_dotenv()


class VecCache:
    """A caching class for vectors and their associated texts.

    Attributes:
        ttl (int): Time-to-live in seconds.
        embedding_model_name (str): The name of the embedding model to be used.
        index (faiss.IndexFlatL2): FAISS index for fast nearest neighbor search.
        texts (List[StoredText]): A list to store the texts.
        distance_thresh (float): The distance threshold for vector search.
    """

    def __init__(
        self,
        ttl: int,
        openai_api_key: str = "",
        embedding_model_name: str = "text-embedding-ada-002",
        vector_size: int = 1536,
        distance_thresh: float = 0.05,
    ):
        """Initializes the VecCache class with provided configurations.

        Args:
            ttl (int): Time-to-live in seconds.
            openai_api_key (str): OpenAI API key. Default is an empty string.
            embedding_model_name (str): Name of the embedding model. Default is "text-embedding-ada-002".
            vector_size (int): Size of the vector. Default is 1536.
            distance_thresh (float): Distance threshold for vector search. Default is 0.05.
        """
        self.ttl = ttl
        self.embedding_model_name = embedding_model_name
        self.index = self._setup_db(vector_size)
        self.texts: List[StoredText] = []
        self.distance_thresh = distance_thresh
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY", "")

    def _setup_db(self, size: int) -> faiss.IndexFlatL2:
        """Sets up the FAISS index.

        Args:
            size (int): Size of the vector.

        Returns:
            faiss.IndexFlatL2: FAISS index object.
        """
        return faiss.IndexFlatL2(size)

    def _add_vector(self, text: str, vector: List[float]) -> None:
        """Adds the text and its associated vector to the cache.

        Args:
            text (str): The text string.
            vector (List[float]): The associated vector of the text.
        """
        self.index.add(np.array([vector], dtype="float32"))
        self.texts += [StoredText(text=text)]

    def _generate_vector(self, text: str) -> List[float]:
        """Generates a vector for the given text using OpenAI's Embedding.

        Args:
            text (str): The text string.

        Returns:
            List[float]: The generated vector.
        """
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=self.embedding_model_name)[
            "data"
        ][0]["embedding"]

    def store(self, text: str) -> None:
        """Stores the given text and its associated vector to the cache.

        Args:
            text (str): The text string.
        """
        vector = self._generate_vector(text)
        self._add_vector(text, vector)

    def store_with_vector(self, text: str, vector: List[float]) -> None:
        """Stores the given text and its associated vector to the cache without generating a new vector.

        Args:
            text (str): The text string.
            vector (List[float]): The vector associated with the text.
        """
        self._add_vector(text, vector)

    def _vector_search(
        self, vector: List[float], return_with_distance: bool
    ) -> Union[str, Tuple[str, float]]:
        """Searches for the closest text in the cache for the given vector.

        Args:
            vector (List[float]): The input vector.
            return_with_distance (bool): Whether to return the distance alongside the text.

        Returns:
            Union[str, Tuple[str, float]]: The closest text or a tuple containing the text and its distance.
        """
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
        """Searches for the closest text in the cache for the given text.

        Args:
            text (str): The input text.
            return_with_distance (bool): Whether to return the distance alongside the text. Default is False.

        Returns:
            Union[str, Tuple[str, float]]: The closest text or a tuple containing the text and its distance.
        """
        vector = self._generate_vector(text)
        return self._vector_search(vector, return_with_distance)

    def search_with_vector(
        self, vector: List[float], return_with_distance: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """Searches for the closest text in the cache for the given vector without generating a new vector.

        Args:
            vector (List[float]): The input vector.
            return_with_distance (bool): Whether to return the distance alongside the text. Default is False.

        Returns:
            Union[str, Tuple[str, float]]: The closest text or a tuple containing the text and its distance.
        """
        return self._vector_search(vector, return_with_distance)

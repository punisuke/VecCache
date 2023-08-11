import pytest

from vec_cache.core import VecCache


class TestVecCache:
    def setup_method(self):
        # Executed before each test
        self.vec_cache = VecCache(ttl=300, openai_api_key="YOUR_KEY")

    @pytest.fixture
    def mock_generate_vector(self, mocker):
        # Mock _generate_vector to return a dummy vector
        return mocker.patch.object(
            VecCache, "_generate_vector", return_value=[1.0] * 1536
        )

    @pytest.fixture
    def mock_openai_embedding(self, mocker):
        # Mock openai.Embedding.create to return a dummy embedding
        return mocker.patch(
            "openai.Embedding.create",
            return_value={"data": [{"embedding": [1.0] * 1536}]},
        )

    def test_setup_db(self):
        index = self.vec_cache._setup_db(1536)
        assert isinstance(index, type(self.vec_cache.index))

    def test_add_vector(self, mock_generate_vector):
        initial_len = len(self.vec_cache.texts)
        self.vec_cache._add_vector("sample text", [1.0] * 1536)
        assert len(self.vec_cache.texts) == initial_len + 1
        assert self.vec_cache.texts[-1] == "sample text"

    def test_generate_vector(self, mock_openai_embedding):
        vector = self.vec_cache._generate_vector("Hello, World!")
        assert len(vector) == 1536
        assert vector[0] == 1.0  # As per our mock

    def test_store_and_search(self, mock_generate_vector, mock_openai_embedding):
        self.vec_cache.store("Hello, World!")
        result = self.vec_cache.search("Hello, World!")
        assert result == "Hello, World!"

    def test_store_with_vector_and_search(
        self, mock_generate_vector, mock_openai_embedding
    ):
        dummy_vector = [1.0] * 1536
        self.vec_cache.store_with_vector("Hello, World!", dummy_vector)
        result = self.vec_cache.search("Hello, World!", return_with_distance=True)
        assert result[0] == "Hello, World!"

# VecCache

VecCache is a Python package that provides an easy interface to cache embeddings of texts using OpenAI's API and allows for fast similarity searches over the cached embeddings.

## Features

- Utilize OpenAI's API to generate text embeddings.
- Cache the embeddings efficiently using `faiss`.
- Quick similarity searches over the cached embeddings.

## Installation

```bash
pip install vec_cache
```

> Note: You need to have an OpenAI API key to use this package.

## Usage

### Basic Usage

```python
from vec_cache import VecCache

# Initialize the cache with your OpenAI API key. If openai_api_key is not provided, .env or environment variable is loaded.
cache = VecCache(ttl=3600, openai_api_key="YOUR_OPENAI_API_KEY")

# Store a text in the cache
cache.store("Hello, World!")

# Search for the most similar text in the cache
result = cache.search("Hi, World!")
print(result)  # Outputs: "Hello, World!"
```

### Storing with Precomputed Vectors

If you have precomputed vectors, you can store them directly:

```python
vector = [0.1, 0.2, ...]  # Some precomputed vector
cache.store_with_vector("Some text", vector)
```

### Searching with Distance

If you wish to obtain the distance along with the result:

```python
result, distance = cache.search("Hi, World!", return_with_distance=True)
print(result)   # Outputs: "Hello, World!"
print(distance) # Outputs the L2 distance
```

## Configuration

`VecCache` accepts the following parameters:

- `ttl`: Time to live for cached vectors.
- `openai_api_key`: Your OpenAI API key.
- `embedding_model_name`: Name of the OpenAI embedding model. Default is "text-embedding-ada-002".
- `vector_size`: Size of the vector. Default is 1536.

## Contributions

We welcome contributions! Please submit a pull request or open an issue if you'd like to improve the package.

### Using `pre-commit` for a Smoother Contribution Experience

To ensure consistent code quality and avoid common pitfalls, we use `pre-commit` hooks in our development process. Before you submit your pull request, it's recommended that you set up `pre-commit` to ensure your changes adhere to our coding standards and guidelines.

**Steps to set up `pre-commit`:**

1. **Install `pre-commit`**:
   If you haven't already, install packages including `pre-commit`:
   ```bash
   poetry install
   ```

2. **Activate `pre-commit` for the repository**:
   Navigate to the root directory of this repository and run:
   ```bash
   pre-commit install
   ```

## License

This project is licensed under the MIT License.

---

Feel free to customize the content based on your package's details and specific requirements.


Of course! Here's a message you can include in your `README.md` file to encourage contributors to use `pre-commit`:

---

### Contributing to [Your Project Name]

#### Using `pre-commit` for a Smoother Contribution Experience

To ensure consistent code quality and avoid common pitfalls, we use `pre-commit` hooks in our development process. Before you submit your pull request, it's recommended that you set up `pre-commit` to ensure your changes adhere to our coding standards and guidelines.

**Steps to set up `pre-commit`:**

1. **Install `pre-commit`**:
   If you haven't already, install `pre-commit`:
   ```bash
   pip install pre-commit
   ```

2. **Activate `pre-commit` for the repository**:
   Navigate to the root directory of this repository and run:
   ```bash
   pre-commit install
   ```

3. **Run `pre-commit` on all files (optional)**:
   If you want to run the hooks on all the files in the repository (useful for the first-time setup):
   ```bash
   pre-commit run --all-files
   ```

With `pre-commit` set up, before every commit, the hooks will automatically check your changes and ensure they meet the repository's guidelines. If any issues are found, `pre-commit` will notify you, and in some cases, it might auto-fix the problems.

We appreciate your contribution and effort, and by setting up `pre-commit`, you're helping us maintain the quality and consistency of the project. Thank you!

---

Make sure to customize the placeholder `[Your Project Name]` with the actual name of your project. Adjust any other details as needed to better fit the specifics of your project or your desired messaging style.

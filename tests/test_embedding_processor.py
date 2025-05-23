import torch
import torch.nn.functional as F

import pytest
import numpy as np
from typing import List
from nlp.processors import EmbeddingProcessor

class MockEmbeddingProcessor(EmbeddingProcessor):
    """
    A mock implementation of the EmbeddingProcessor for testing purposes.
    """

    def encode(self, text_corpus: List[str], normalize_embeddings: bool = False) -> torch.Tensor:
        """
        Encodes a list of text strings into fixed-size embeddings.
    
        Args:
            text_corpus (List[str]): A list of strings to encode.
            normalize_embeddings (bool, optional): If True, normalize each
                embedding vector to unit length. Defaults to False.

        Returns:
            torch.Tensor: A tensor of shape (len(text_corpus), 10) containing
                mock embeddings.
        """
        embeddings = torch.randn(len(text_corpus), 10, dtype=torch.float32)
        if normalize_embeddings:
            return F.normalize(torch.rand(len(text_corpus), 10, dtype=torch.float32), dim=1)
        else:
            return embeddings

@pytest.fixture
def mock_embedding_processor() -> MockEmbeddingProcessor:
    """
    Fixture for creating a mock embedding processor.
    """
    return MockEmbeddingProcessor()

def test_encode_single_string(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the encode method with a single string input.
    """
    text_corpus = ["This is a test."]
    embeddings = mock_embedding_processor.encode(text_corpus)
    assert embeddings.shape == (1, 10), "The shape of the embeddings is incorrect."

def test_encode_multiple_strings(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the encode method with multiple string inputs.
    """
    text_corpus = ["This is a test.", "This is another test."]
    embeddings = mock_embedding_processor.encode(text_corpus)
    assert embeddings.shape == (2, 10), "The shape of the embeddings is incorrect."

def test_encode_normalize_embeddings(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the encode method with normalization.
    """
    text_corpus = ["This is a test.", "This is another test."]
    embeddings = mock_embedding_processor.encode(text_corpus, normalize_embeddings=True)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(embeddings.size(0))), "the embeddings are not normalized."

def test_cosine_similarity(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the cosine similarity method.
    """
    tensor_one = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor_two = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    expected_result = torch.tensor([0.9734, 0.9972])
    cosine_similarity_tensor = mock_embedding_processor.cosine_similarity(tensor_one, tensor_two)
    print(cosine_similarity_tensor)
    assert torch.allclose(cosine_similarity_tensor, expected_result, atol=1e-2), "The cosine similarity result is incorrect."

def test_cosine_similarity_orthogonal(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the cosine similarity method with orthogonal vectors.
    """
    tensor_one = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tensor_two = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected_result = torch.tensor([0.0, 0.0])
    cosine_similarity_tensor = mock_embedding_processor.cosine_similarity(tensor_one, tensor_two)
    assert torch.allclose(cosine_similarity_tensor, expected_result), "The cosine similarity result is incorrect."

def test_cosine_similarity_negative(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the cosine similarity method with vectors pointing in opposite directions.
    """
    tensor_one = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tensor_two = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
    
    expected_result = torch.tensor([-1.0, -1.0])
    cosine_similarity_tensor = mock_embedding_processor.cosine_similarity(tensor_one, tensor_two)
    assert torch.allclose(cosine_similarity_tensor, expected_result), "The cosine similarity result is incorrect."

def test_dot_product(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the the dot product method.
    """
    tensor_one = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor_two = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected_result = torch.tensor([[17.0, 23.0], [39.0, 53.0]])
    dot_product_tensor = mock_embedding_processor.dot_product(tensor_one, tensor_two)
    assert torch.allclose(dot_product_tensor, expected_result), "The dot product result is incorrect."

def test_dot_product_identity(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the dot product method with identity matrices.
    """
    tensor_one = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tensor_two = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    expected_result = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dot_product_tensor = mock_embedding_processor.dot_product(tensor_one, tensor_two)
    assert torch.allclose(dot_product_tensor, expected_result), "The dot product result is incorrect."

def test_dot_product_zeros(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the dot product method with zero tensors.
    """
    tensor_one = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    tensor_two = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_result = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    dot_product_tensor = mock_embedding_processor.dot_product(tensor_one, tensor_two)
    assert torch.allclose(dot_product_tensor, expected_result), "The dot product result is incorrect."

def test_to_numpy(mock_embedding_processor: MockEmbeddingProcessor) -> None:
    """
    Test the to_numpy method.
    """
    torch_tensor = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    numpy_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    output = mock_embedding_processor.to_numpy(torch_tensor)
    assert np.array_equal(output, numpy_tensor), "The output is not equal to the expected NumPy array."
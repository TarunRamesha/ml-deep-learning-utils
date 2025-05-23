import pytest
import torch
from nlp.utils import mean_pooling_pt, max_pooling_pt  # Replace with the actual import path

def test_mean_pooling_basic():
    """
    Test mean pooling on embeddings where all tokens are valid.
    Verifies the average is computed over all token embeddings.
    """
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[1, 1]])
    result = mean_pooling_pt(embeddings, mask)
    expected = torch.tensor([[2.0, 3.0]])
    assert torch.allclose(result, expected, atol=1e-6)

def test_mean_pooling_with_padding():
    """
    Test mean pooling with padding.
    Verifies that only tokens with attention mask of 1 are included in the average.
    """
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[1, 0]])
    result = mean_pooling_pt(embeddings, mask)
    expected = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(result, expected, atol=1e-6)

def test_mean_pooling_shape_mismatch():
    """
    Test mean pooling with mismatched shapes between embeddings and attention mask.
    Verifies that a ValueError is raised.
    """
    embeddings = torch.rand(2, 3, 4)
    mask = torch.ones(2, 4)  # incorrect shape
    with pytest.raises(ValueError, match="shape of `attention_mask` must match"):
        mean_pooling_pt(embeddings, mask)

def test_max_pooling_basic():
    """
    Test max pooling on embeddings where all tokens are valid.
    Verifies the max value is taken across the sequence dimension.
    """
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 0.0]]])
    mask = torch.tensor([[1, 1]])
    result = max_pooling_pt(embeddings.clone(), mask)
    expected = torch.tensor([[3.0, 2.0]])
    assert torch.allclose(result, expected, atol=1e-6)

def test_max_pooling_with_padding():
    """
    Test max pooling with padding.
    Verifies that padding tokens are masked with a large negative value and ignored.
    """
    embeddings = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 0]])
    result = max_pooling_pt(embeddings.clone(), mask)
    expected = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(result, expected, atol=1e-6)

def test_max_pooling_shape_mismatch():
    """
    Test max pooling with mismatched shapes between embeddings and attention mask.
    Verifies that a ValueError is raised.
    """
    embeddings = torch.rand(3, 5, 8)
    mask = torch.ones(3, 4)  # incorrect shape
    with pytest.raises(ValueError, match="shape of `attention_mask` must match"):
        max_pooling_pt(embeddings, mask)

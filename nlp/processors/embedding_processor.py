import torch
import torch.nn.functional as F

import numpy as np
from typing import List, Union
from abc import ABC, abstractmethod

class EmbeddingProcessor(ABC):
    """
    Abstract base class for embedding processors.
    """

    @abstractmethod
    def encode(self, text_corpus: List[str], normalize_embeddings: bool = False) -> torch.Tensor:
        """
        Encodes the input text corpus into embeddings.

        Args:
            text_corpus (List[str]]): List of texts to encode.
            normalize_embeddings (bool): Whether to normalize the embeddings.

        Returns:
            torch.Tensor: The encoded embeddings.
        """
        raise NotImplementedError("Subclasses must implement this method.") 

    def cosine_similarity(self, tensor_one: torch.Tensor, tensor_two: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity between two tensors.

        Args:
            tensor_one (torch.Tensor): The first tensor.
            tensor_two (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: The cosine similarity between the two tensors.
        """
        return F.cosine_similarity(tensor_one, tensor_two, dim=1)
    
    def dot_product(self, tensor_one: torch.Tensor, tensor_two: torch.Tensor) -> torch.Tensor:
        """
        Computes the dot product between two tensors.

        Args:
            tensor_one (torch.Tensor): The first tensor.
            tensor_two (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: The dot product between the two tensors.
        """
        return torch.matmul(tensor_one, tensor_two.T)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a PyTorch tensor to a NumPy array.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            np.ndarray: The converted NumPy array.
        """
        return tensor.cpu().detach().numpy() if tensor.is_cuda else tensor.detach().numpy()
import random
import pytest
from unittest.mock import MagicMock, patch

from nlp.utils import truncate_text

RANDOM_WORDS = [
    "apple", "banana", "cat", "dog", "elephant", "forest", "giraffe",
    "house", "ice", "jungle", "kite", "lemon", "mountain", "night", "ocean",
    "penguin", "queen", "river", "sun", "tree", "umbrella", "violin",
    "whale", "xray", "yacht", "zebra"
]

def _generate_random_text_body(word_count: int) -> str:
    """
    Generate a text body of random words for testing purposes.

    Args:
        word_count (int): The number words to include in the text body.

    Returns:
        str: Random text body of size word count.  
    """
    return ' '.join(random.choices(RANDOM_WORDS, k=word_count))

@pytest.fixture
def mock_text_corpus():
    """
    Pytest fixture that provides a function to generate a mock text corpus.

    Returns:
        Callable (str, int): A function that takes a word count (int) and 
        returns a string of randomly generated words of that length.
    """
    def _generate(word_count: int):
        return _generate_random_text_body(word_count)
    return _generate

@pytest.fixture
def mock_tokenizer():
    """
    Pytest fixture that provides a MagicMock simulating a basic tokenizer.

    The mock object includes:
    - `tokenize(text)`: Simulates tokenization by splitting the input string on whitespace.
    - `convert_tokens_to_string(tokens)`: Simulates converting tokens back into a string 
       by joining them with spaces.

    Returns:
        MagicMock: A mock tokenizer object mimicking basic tokenizer behavior.
    """
    mock = MagicMock()
    mock.tokenize.side_effect = lambda x: x.split()
    mock.convert_tokens_to_string.side_effect = lambda tokens: ' '.join(tokens)
    return mock

@patch("nlp.utils.preprocess..AutoTokenizer.from_pretrained")
def test_truncate_text_exact_limit(mock_from_pretrained, mock_tokenizer):
    """
    Test that the input text is returned unchanged when the number of tokens
    exactly matches the token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = "This is a test string."
    token_limit = 5

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == input_text

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_below_limit(mock_from_pretrained, mock_tokenizer):
    """
    Test that the input text is returned unchanged when the number of tokens
    is below the specified token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = "Short text"
    token_limit = 10

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == input_text

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_above_limit(mock_from_pretrained, mock_tokenizer):
    """
    Test that the text is truncated correctly when the number of tokens
    exceeds the specified token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = "This is a test string with more than five tokens."
    token_limit = 5

    result = truncate_text(input_text, token_limit, "fake-model")
    expected = "This is a test string"
    assert result == expected

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_empty_input(mock_from_pretrained, mock_tokenizer):
    """
    Test that an empty string is returned when the input text is empty,
    regardless of the token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = ""
    token_limit = 10

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == ""

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_zero_limit(mock_from_pretrained, mock_tokenizer):
    """
    Test that an empty string is returned when the token limit is set to zero,
    regardless of the input text.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = "Any non-empty input"
    token_limit = 0

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == ""

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_empty_input(mock_from_pretrained, mock_tokenizer):
    """
    Test that an empty string is returned when the input text is empty,
    regardless of the token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = ""
    token_limit = 10

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == ""

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_exact_limit(mock_from_pretrained, mock_tokenizer):
    """
    Test that the input text is returned unchanged when the number of tokens
    exactly matches the token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = "This is a test string."
    token_limit = 5

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == input_text

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_below_limit(mock_from_pretrained, mock_tokenizer):
    """
    Test that the input text is returned unchanged when the number of tokens
    is below the specified token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = "Short text"
    token_limit = 10

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == input_text

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_above_limit(mock_from_pretrained, mock_tokenizer):
    """
    Test that the text is truncated correctly when the number of tokens
    exceeds the specified token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = "This is a test string with more than five tokens."
    token_limit = 5

    result = truncate_text(input_text, token_limit, "fake-model")
    expected = "This is a test string"
    assert result == expected

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_empty_input(mock_from_pretrained, mock_tokenizer):
    """
    Test that an empty string is returned when the input text is empty,
    regardless of the token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = ""
    token_limit = 10

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == ""

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_zero_limit(mock_from_pretrained, mock_tokenizer):
    """
    Test that an empty string is returned when the token limit is set to zero,
    regardless of the input text.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = "Any non-empty input"
    token_limit = 0

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == ""

@patch("nlp.utils.preprocess.AutoTokenizer.from_pretrained")
def test_truncate_text_empty_input(mock_from_pretrained, mock_tokenizer):
    """
    Test that an empty string is returned when the input text is empty,
    regardless of the token limit.
    """
    mock_from_pretrained.return_value = mock_tokenizer
    input_text = ""
    token_limit = 10

    result = truncate_text(input_text, token_limit, "fake-model")
    assert result == ""

def test_whitespace_tokenization_exact_limit():
    """
    Test truncation using whitespace tokenization when input has exactly the token limit.
    """
    input_text = "one two three"
    token_limit = 3
    result = truncate_text(input_text, token_limit)
    assert result == input_text

def test_whitespace_tokenization_below_limit():
    """
    Test truncation using whitespace tokenization when input has fewer tokens than the limit.
    """
    input_text = "just one"
    token_limit = 5
    result = truncate_text(input_text, token_limit)
    assert result == input_text

def test_whitespace_tokenization_above_limit():
    """
    Test that input is correctly truncated when it exceeds the token limit.
    """
    input_text = "this input should be truncated to only four words"
    token_limit = 4
    result = truncate_text(input_text, token_limit)
    assert result == "this input should be"

def test_whitespace_tokenization_zero_limit():
    """
    Test that an empty string is returned when the token limit is zero.
    """
    input_text = "this should return nothing"
    token_limit = 0
    result = truncate_text(input_text, token_limit)
    assert result == ""

def test_whitespace_tokenization_empty_input():
    """
    Test that an empty string is returned when the input is empty.
    """
    input_text = ""
    token_limit = 5
    result = truncate_text(input_text, token_limit)
    assert result == ""
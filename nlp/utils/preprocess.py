from transformers import AutoTokenizer

from typing import Optional
from transformers import AutoTokenizer

def truncate_text(input_text: str, token_limit: int, model_name: Optional[str] = None) -> str:
    """
    Truncate the input text to a specified number of tokens.

    If `model_name` is provided, uses a tokenizer from the specified model.
    Otherwise, falls back to simple whitespace-based tokenization.

    Args:
        input_text (str): The input text to be truncated.
        token_limit (int): The maximum number of tokens allowed.
        model_name (Optional[str]): The name of the model to use for tokenization.
            If None, basic whitespace splitting is used.

    Returns:
        str: The truncated text.
    """
    if model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(input_text)
        if len(tokens) > token_limit:
            tokens = tokens[:token_limit]
            return tokenizer.convert_tokens_to_string(tokens)
        return input_text
    else:
        tokens = input_text.split()
        if len(tokens) > token_limit:
            return ' '.join(tokens[:token_limit])
        return input_text
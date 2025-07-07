"""
Utility functions for Phase 2.2 pile activation caching.
"""

from typing import Optional
import torch


def find_word_position(word: str, input_ids: torch.Tensor, tokenizer) -> Optional[int]:
    """
    Find the position of a word in the tokenized sequence.
    
    This function handles the fact that a word might be split across multiple tokens
    by checking if the word appears when decoding up to each position.
    
    Args:
        word: The word to find
        input_ids: Tensor of token IDs [seq_len]
        tokenizer: The tokenizer used to encode the text
        
    Returns:
        Position of the first token that completes the word, or None if not found
    """
    # Handle empty word or empty sequence
    if not word or len(input_ids) == 0:
        return None
        
    # Convert word to lowercase for case-insensitive matching
    word_lower = word.lower()
    
    # Check each position to see if the word appears when decoded up to that point
    for pos in range(len(input_ids)):
        # Decode tokens up to and including this position
        decoded = tokenizer.decode(input_ids[:pos+1])
        decoded_lower = decoded.lower()
        
        # Check if the word appears in the decoded text
        if word_lower in decoded_lower:
            # Make sure this is the first occurrence by checking previous position
            if pos == 0:
                return pos
            else:
                prev_decoded = tokenizer.decode(input_ids[:pos]).lower()
                if word_lower not in prev_decoded:
                    return pos
    
    return None


def validate_pile_sample(text: str, word: str, max_length: int = 128) -> bool:
    """
    Validate that a pile sample is suitable for processing.
    
    Args:
        text: The full text sample
        word: The random word selected
        max_length: Maximum token length for truncation
        
    Returns:
        True if the sample is valid for processing
    """
    # Check basic validity
    if not text or not word:
        return False
        
    # Check if word exists in text (case-insensitive)
    if word.lower() not in text.lower():
        return False
        
    # Additional validation can be added here
    # For now, we rely on the tokenization check in the main runner
    
    return True
import random
import numpy as np


def sample_span_lenghts(num_tokens : int, corruption_rate : float = 0.15, mean_span_lenght : int = 3):
    """
    Sample span lenghts that corrupt approximately corrpution_rate of tokens.
    Use geometric distribution with a specified mean
    Taken from: https://mbrenndoerfer.com/writing/t5-pretraining-span-corruption-denoising-objectives#t5-pre-training
    """

    # Calculate expected number of tokens to corrput
    
    number_to_corrupt = int(num_tokens * corruption_rate) # 15% of the tokens should be corrupted

    p = 1.0 / mean_span_lenght # The probability divided to the mean (1 / 3 == 0.33)

    spans = []
    total_corrupted = 0

    while total_corrupted < number_to_corrupt:
        # Sample from geometric distribution (minimum lenght 1)
        span_lenght = np.random.geometric(p)

        # Dont exceed the remaining budget 
        span_lenght = min(span_lenght, number_to_corrupt - total_corrupted)
        spans.append(span_lenght)
        total_corrupted += span_lenght
    return spans
import random
import numpy as np
from typing import Optional

def corrupt_spans(tokens, sentinel_ids, eos_id,  corruption_rate : float = 0.15, mean_span_lenght : int = 3, seed : Optional[int] = None):
    """
    Apply span corruption to a list of tokens

    Returns:
        corrupted_input : tokens with spans replaced by sentinels
        target: sentinel tokens followed by their corresponding spans
        Taken from: https://mbrenndoerfer.com/writing/t5-pretraining-span-corruption-denoising-objectives#t5-pre-training 
    """

    if seed is not None:
        np.random.seed(seed=seed)
        random.seed(seed)

    num_tokens = len(tokens)
    span_lenghts = sample_span_lenghts(num_tokens=num_tokens, corruption_rate=corruption_rate, mean_span_lenght=mean_span_lenght)

    # Randomly places spans (non-overlapping)
    # Create a list of possible start positions
    available_positions = set(range(num_tokens))
    spans_positions = [] # List of (start, lenght) tuples

    for lenght in span_lenghts:
        # Filter position where span would fit without overlap
        valid_starts = []
        for pos in available_positions:
            # Check if span fits and doesnt overlap existing spans
            if pos + lenght <= num_tokens:
                if all(positions in available_positions for positions in range(pos, pos + lenght)): #Checkin the positions directly
                    valid_starts.append(pos)

        if not valid_starts:
            continue


        # Randomly select start position
        start = random.choice(valid_starts)
        spans_positions.append((start, lenght))

        #Remove these positions from avaible positions so it doesnt overlap
        for index in range(start, start + lenght):
            available_positions.remove(index)

    #Sort spans by position for consisten ordering
    spans_positions.sort(key=lambda x:x[0])

    #Build corrupted input and target
    corrupted_input = []
    target = []
    sentinel_id = 0
    current_pos = 0
        
    for start, lenght in spans_positions:
        # Add tokens before this span
        corrupted_input.extend(tokens[current_pos:start])

        #Add a sentinel token
        corrupted_input.append(sentinel_ids[sentinel_id])

        #Add to target: sentinel followed by span content
        target.append(sentinel_ids[sentinel_id])
        target.extend(tokens[start : start + lenght])

        sentinel_id += 1
        current_pos = start + lenght

    # Add remaining tokens after last span
    corrupted_input.extend(tokens[current_pos:])
    corrupted_input.append(eos_id)
    target.append(eos_id)
    return corrupted_input, target
            

def sample_span_lenghts(num_tokens : int, corruption_rate : float = 0.15, mean_span_lenght : int = 3):
    """
    Sample span lenghts that corrupt approximately corrpution_rate of tokens.
    Use geometric distribution with a specified mean
    Taken from: https://mbrenndoerfer.com/writing/t5-pretraining-span-corruption-denoising-objectives#t5-pre-training
    """

    # Calculate expected number of tokens to corrput
    
    number_to_corrupt = max(1, int(num_tokens * corruption_rate)) # 15% of the tokens should be corrupted

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


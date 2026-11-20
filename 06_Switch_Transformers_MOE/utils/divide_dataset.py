from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset
    
def divide_dataset_train_val_test(dataset : DatasetDict, test_size : float = 0.5, seed : int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Divide the dateset into train, validation and test sets.
    """
    train_ds = dataset["train"]
    
    # Split the validation set into validation and test sets
    tmp = dataset["validation"].train_test_split(test_size=test_size, seed=seed)
    val_ds, test_ds = tmp["train"], tmp["test"]
    

    return train_ds, val_ds, test_ds
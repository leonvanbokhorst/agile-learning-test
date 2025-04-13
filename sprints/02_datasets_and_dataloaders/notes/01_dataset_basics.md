# PyTorch Dataset Basics

## The Dataset Class

PyTorch's `Dataset` class is the foundation for handling data. It's an abstract class that requires two main methods:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, ...):
        # Initialize your dataset
        # Load data, set transforms, etc.
        pass

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single sample at index idx
        # This is where preprocessing happens
        return self.data[idx]
```

## Key Concepts

1. **Initialization (`__init__`)**:

   - Load data into memory or set up paths
   - Initialize transforms
   - Set up any necessary preprocessing

2. **Length (`__len__`)**:

   - Must return the total number of samples
   - Used by DataLoader to know when to stop

3. **Item Access (`__getitem__`)**:
   - Returns a single sample
   - Can include preprocessing
   - Should return a tuple of (input, target)

## Example: Simple Dataset

```python
class SimpleDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target
```

## Best Practices

1. **Memory Management**:

   - Load data lazily if possible
   - Use transforms for preprocessing
   - Consider using memory mapping for large datasets

2. **Preprocessing**:

   - Keep transforms separate from data loading
   - Use composition for complex transforms
   - Consider caching preprocessed data

3. **Error Handling**:
   - Handle missing or corrupted data
   - Provide meaningful error messages
   - Consider data validation

## Common Pitfalls

1. **Memory Issues**:

   - Loading entire dataset into memory
   - Not using proper batching
   - Ignoring worker processes

2. **Performance**:

   - Inefficient data loading
   - Unnecessary preprocessing
   - Poor transform pipeline design

3. **Data Consistency**:
   - Inconsistent preprocessing
   - Missing data handling
   - Incorrect indexing

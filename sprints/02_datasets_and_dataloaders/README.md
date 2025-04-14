# Sprint 2: Datasets & DataLoaders

**Parent Epic:** Master PyTorch Fundamentals and Construct a Foundational Transformer LLM (See `../../README.md`)

## Sprint Goal

Learn how to efficiently load, preprocess, and manage data using PyTorch's `Dataset` and `DataLoader` classes. This will form the foundation for all future training tasks.

## Tasks / Learning Objectives

1. **Understanding PyTorch Dataset:**

   - [ ] Study the `Dataset` class interface
   - [ ] Implement a custom dataset for a simple task
   - [ ] Understand `__len__` and `__getitem__` methods
   - [ ] Handle data loading and preprocessing

2. **Working with DataLoaders:**

   - [x] Configure batch size and shuffling
   - [x] Implement parallel data loading with workers
   - [ ] Handle different data types (images, text, etc.)
   - [ ] Understand memory management

3. **Data Transformation Pipeline:**

   - [ ] Implement data augmentation
   - [ ] Create custom transforms
   - [ ] Use torchvision.transforms
   - [ ] Handle normalization and preprocessing

4. **Built-in Datasets:**
   - [x] Explore torchvision.datasets
   - [ ] Work with torchtext.datasets
   - [ ] Understand dataset splits (train/val/test)
   - [ ] Handle dataset downloading and caching

## Definition of Done / Key Questions Answered

- [ ] Can create custom datasets for different data types
- [ ] Understand how to efficiently load and preprocess data
- [ ] Can implement data augmentation and transformation pipelines
- [ ] Know how to use built-in datasets and create custom ones
- [ ] Key Question: How does PyTorch handle data loading and preprocessing efficiently?

## Expected Outcomes

1. **Code Examples:**

   - Custom dataset implementation
   - DataLoader configuration
   - Transformation pipeline
   - Working with built-in datasets

2. **Documentation:**
   - Notes on dataset creation
   - Best practices for data loading
   - Memory management considerations
   - Performance optimization tips

## Prerequisites

- Completion of Sprint 1 (Setup & Basics)
- Understanding of PyTorch tensors and basic operations
- Familiarity with Python classes and inheritance

## Next Steps

After completing this sprint, we'll move on to:

- Building more complex neural networks
- Implementing transformer components
- Setting up training pipelines

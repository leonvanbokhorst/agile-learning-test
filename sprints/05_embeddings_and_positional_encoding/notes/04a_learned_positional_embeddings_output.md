# Learned Positional Embeddings Output

Let's break down the output from the `sprints/05_embeddings_and_positional_encoding/results/learned_pe_example.py` file.

```terminal
Word Embedding Layer: Embedding(100, 16)

Position Embedding Layer: Embedding(50, 16)

Input Word Indices (Shape: torch.Size([4, 10])):
tensor([[20, 73, 40, 25, 14, 87, 65, 62,  7, 21],
        [91, 84, 68, 69,  0, 25, 57, 54,  7, 84],
        [13, 69, 19, 91,  2, 59, 54, 23, 45, 65],
        [65, 65, 72, 83, 92, 18, 12, 92, 28, 72]])

Position Indices (Shape: torch.Size([4, 10])):
tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

Word Embeddings (Shape: torch.Size([4, 10, 16]))

Positional Embeddings (Shape: torch.Size([4, 10, 16]))

Final Combined Embeddings (Shape: torch.Size([4, 10, 16]))
```

1.  **`Word Embedding Layer: Embedding(100, 16)`**
    *   **What it is:** This is the computer setting up a "dictionary" for words. It says, "Okay, I expect maybe 100 different words in total (`num_embeddings=100`). For each word, I'm going to learn a secret code that's 16 numbers long (`embedding_dim=16`)."
    *   **Why:** This code helps the computer understand the *meaning* or *context* of a word. Initially, the codes are random, but they get better as the computer learns.

2.  **`Position Embedding Layer: Embedding(50, 16)`**
    *   **What it is:** Similar setup, but this dictionary is for *positions* in a sentence, not words. "I need to handle sentences up to 50 words long (`num_embeddings=max_seq_len=50`). For each position (1st, 2nd, 3rd... up to 50th), I'll learn a *separate* secret code, also 16 numbers long (`embedding_dim=16`)."
    *   **Why:** This tells the computer *where* a word is in the sentence. Position 0 gets one code, position 1 gets another, and so on. Again, these codes are *learned*.

3.  **`Input Word Indices (Shape: torch.Size([4, 10]))`**
    *   **What it is:** This is our pretend data. Imagine 4 short sentences (`batch_size=4`), each with 10 words (`seq_len=10`). The numbers you see (like 44, 93, 4...) are just numerical IDs representing words from our 100-word vocabulary. So, '44' might be 'the', '93' might be 'quick', etc.
    *   **Shape `[4, 10]`:** 4 rows (sentences), 10 columns (words per sentence).

4.  **`Position Indices (Shape: torch.Size([4, 10]))`**
    *   **What it is:** This simply lists the position of each word in our pretend sentences. Since each sentence has 10 words, the positions are always 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. It's repeated 4 times, once for each sentence.
    *   **Why:** The computer uses *these* numbers (0-9) to look up the corresponding 16-number codes from the `Position Embedding Layer`.

5.  **`Word Embeddings (Shape: torch.Size([4, 10, 16]))`**
    *   **What it is:** Here, the computer looked up the 16-number code for each of the word IDs in our input.
    *   **Shape `[4, 10, 16]`:** We still have 4 sentences, 10 words each, but now each word isn't just an ID (like 44), it's represented by its full 16-number code (embedding vector).

6.  **`Positional Embeddings (Shape: torch.Size([4, 10, 16]))`**
    *   **What it is:** Same process, but using the `Position Indices` (0-9) and the `Position Embedding Layer`. Now, each *position* in each sentence has its learned 16-number code.
    *   **Shape `[4, 10, 16]`:** 4 sentences, 10 positions each, and each position is represented by its 16-number code.

7.  **`Final Combined Embeddings (Shape: torch.Size([4, 10, 16]))`**
    *   **What it is:** This is the grand finale! We simply **added** the `Word Embeddings` and the `Positional Embeddings` together, element by element. The 1st number of the word code gets added to the 1st number of the position code, the 2nd to the 2nd, and so on, for all 16 numbers.
    *   **Why:** The resulting 16-number code for each spot now contains information about *both* the word itself *and* its position in the sentence. This combined vector is what usually gets fed into the next part of a Transformer model.
    *   **Shape `[4, 10, 16]`:** The shape doesn't change because we just added corresponding numbers together.

**In short:** We took word IDs, looked up their learned "meaning" codes. We took position numbers, looked up their learned "location" codes. Then, we just added the two codes together for each word/position pair to get a combined representation. Voila! Learned Positional Embeddings in action.

# Direct Reversal Demo Script

import torch
from train_seq_reversal import (
    generate_sequence_pair,
    MIN_SEQ_LEN,
    MAX_SEQ_LEN,
    MIN_NUM,
    MAX_NUM,
    BOS_TOKEN,
    EOS_TOKEN,
)


def direct_reverse(seq):
    """Return the reverse of a sequence of integers."""
    return list(torch.flip(torch.tensor(seq), dims=[0]).tolist())


def main():
    print("Direct Reversal Demo")
    print("=~" * 20)
    for i in range(5):
        src, tgt = generate_sequence_pair(MIN_SEQ_LEN, MAX_SEQ_LEN, MIN_NUM, MAX_NUM)
        src_tokens = [BOS_TOKEN] + src
        expected = tgt + [EOS_TOKEN]
        direct_rev = direct_reverse(src) + [EOS_TOKEN]

        print(f"Sample {i+1}:")
        print(f"  Source tokens:      {src_tokens}")
        print(f"  Expected rev + EOS: {expected}")
        print(f"  Direct reverse:     {direct_rev}")
        print()


if __name__ == "__main__":
    main()

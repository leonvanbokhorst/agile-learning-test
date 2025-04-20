import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import math
from torch.distributions import Categorical

# Assuming the model is in the same directory
from encoder_decoder_model import EncoderDecoder

# --- Configuration --- #
# Vocabulary
PAD_TOKEN = 0
BOS_TOKEN = 1  # Start Of Sequence
EOS_TOKEN = 2  # End Of Sequence
NUM_TOKENS_START = 3  # 0, 1, 2 are special
MIN_NUM = 3
MAX_NUM = 9  # Digits 3 through 9
VOCAB_SIZE = MAX_NUM + 1  # 0=PAD, 1=BOS, 2=EOS, 3-9=digits

# Data Generation
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 12  # Max length of numbers sequence (excluding BOS/EOS)
NUM_SAMPLES = 1000
BATCH_SIZE = 64

# Model Hyperparameters (match model definition defaults where possible)
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
D_MODEL = 128
NUM_HEADS = 8
D_FF = D_MODEL * 4
DROPOUT_PROB = 0.1

# Training Hyperparameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add RL hyperparameter
RL_EPOCHS = 30  # Number of epochs for REINFORCE

print(f"Using device: {DEVICE}")
print(f"Vocabulary Size: {VOCAB_SIZE}")

# --- Data Generation & Dataset --- #


def generate_sequence_pair(min_len, max_len, min_num, max_num):
    """Generates a sequence of numbers and its reverse."""
    seq_len = random.randint(min_len, max_len)
    sequence = [random.randint(min_num, max_num) for _ in range(seq_len)]
    reversed_sequence = sequence[::-1]
    return sequence, reversed_sequence


class SequenceReversalDataset(Dataset):
    """Dataset for the sequence reversal task with dynamic curriculum length."""

    def __init__(
        self,
        num_samples,
        min_len,
        max_len,
        min_num,
        max_num,
        pad_token,
        bos_token,
        eos_token,
    ):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len  # Will be updated per epoch for curriculum
        self.min_num = min_num
        self.max_num = max_num
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a new sequence pair based on current max_len
        src_seq, tgt_seq = generate_sequence_pair(
            self.min_len, self.max_len, self.min_num, self.max_num
        )
        # Add BOS and pad
        src_tokens = [self.bos_token] + src_seq
        src_padded = src_tokens + [self.pad_token] * (
            self.max_len + 1 - len(src_tokens)
        )

        tgt_input_tokens = [self.bos_token] + tgt_seq
        tgt_input_padded = tgt_input_tokens + [self.pad_token] * (
            self.max_len + 1 - len(tgt_input_tokens)
        )

        tgt_output_tokens = tgt_seq + [self.eos_token]
        tgt_output_padded = tgt_output_tokens + [self.pad_token] * (
            self.max_len + 1 - len(tgt_output_tokens)
        )

        return {
            "src": torch.tensor(src_padded, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_input_padded, dtype=torch.long),
            "tgt_output": torch.tensor(tgt_output_padded, dtype=torch.long),
        }


# --- Main Training Logic (to be added) --- #


def evaluate_model(model, num_eval_samples=5, device=DEVICE):
    """Evaluates the model on a few random samples using greedy decoding."""
    print("\n--- Starting Evaluation ---")
    model.eval()  # Set model to evaluation mode

    # Simple mapping for printing (adjust if VOCAB changes)
    token_map = {0: "PAD", 1: "BOS", 2: "EOS"}
    token_map.update({i: str(i) for i in range(MIN_NUM, MAX_NUM + 1)})

    def decode_tokens(tokens):
        return " ".join([token_map.get(tok.item(), "?") for tok in tokens])

    correct_reversals = 0
    with torch.no_grad():
        for i in range(num_eval_samples):
            print(f"\n--- Sample {i+1} ---")
            # 1. Generate a sample
            src_seq, tgt_seq = generate_sequence_pair(
                MIN_SEQ_LEN, MAX_SEQ_LEN, MIN_NUM, MAX_NUM
            )
            expected_output_seq = tgt_seq + [EOS_TOKEN]

            # 2. Prepare source input
            src_tokens_list = [BOS_TOKEN] + src_seq
            src_padded_list = src_tokens_list + [PAD_TOKEN] * (
                MAX_SEQ_LEN + 1 - len(src_tokens_list)
            )
            src_tensor = torch.tensor([src_padded_list], dtype=torch.long).to(device)

            print(
                f"Source Sequence:      {decode_tokens(torch.tensor(src_tokens_list))}"
            )
            print(
                f"Expected Rev + EOS:   {decode_tokens(torch.tensor(expected_output_seq))}"
            )

            # 3. Autoregressive Decoding
            generated_tokens = [BOS_TOKEN]
            max_output_len = MAX_SEQ_LEN + 1  # Max length for generated sequence

            # Get encoder output once
            encoder_output = model.encoder(
                src_tensor, model._create_padding_mask(src_tensor)
            )

            for _ in range(max_output_len):
                # Prepare decoder input tensor
                tgt_input_tensor = torch.tensor(
                    [generated_tokens], dtype=torch.long
                ).to(device)

                # Create masks
                target_mask = model._create_look_ahead_mask(
                    tgt_input_tensor.size(1), device
                )
                # Padding mask for target input usually not needed here if we don't pad `generated_tokens` yet
                # combined_target_mask = target_mask # Simplified

                # Pass through decoder
                decoder_output = model.decoder(
                    target=tgt_input_tensor,
                    encoder_output=encoder_output,
                    target_mask=target_mask,  # Pass look-ahead
                    encoder_mask=model._create_padding_mask(
                        src_tensor
                    ),  # Source padding mask
                )

                # Get logits for the last token
                last_token_logits = model.final_linear(
                    decoder_output[:, -1, :]
                )  # Shape: (1, vocab_size)

                # Greedy decoding: pick token with highest probability
                next_token = last_token_logits.argmax(dim=-1).item()
                generated_tokens.append(next_token)

                # Stop if EOS is generated
                if next_token == EOS_TOKEN:
                    break

            # Remove BOS token for comparison
            generated_sequence_no_bos = generated_tokens[1:]
            print(
                f"Generated Sequence:   {decode_tokens(torch.tensor(generated_tokens))}"
            )
            print(
                f"Generated (no BOS):   {decode_tokens(torch.tensor(generated_sequence_no_bos))}"
            )

            # 4. Compare
            # Pad generated sequence to the same length as expected for comparison
            len_diff = len(expected_output_seq) - len(generated_sequence_no_bos)
            padded_generated = generated_sequence_no_bos + [PAD_TOKEN] * max(
                0, len_diff
            )
            padded_expected = expected_output_seq + [PAD_TOKEN] * max(0, -len_diff)
            # Take min length to avoid comparing extra padding
            compare_len = min(len(padded_generated), len(padded_expected))

            if padded_generated[:compare_len] == padded_expected[:compare_len]:
                print("Result: CORRECT")
                correct_reversals += 1
            else:
                print("Result: INCORRECT")

    print(
        f"\nEvaluation Complete. Correct Reversals: {correct_reversals}/{num_eval_samples}"
    )


# --- Teacher-Forcing Evaluation ---
def teacher_forcing_eval(model, dataloader, device):
    """Evaluate the model under teacher forcing (feeding true previous tokens)."""
    model.eval()
    total_seqs = 0
    correct_seqs = 0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_output = batch["tgt_output"].to(device)
            # Forward under teacher forcing
            logits = model(src, tgt_input)  # (batch, seq_len, vocab)
            preds = logits.argmax(dim=-1)  # (batch, seq_len)
            # Sequence-level correctness (ignore padding)
            mask = tgt_output != PAD_TOKEN
            seq_ok = ((preds == tgt_output) | (~mask)).all(dim=1)
            correct_seqs += seq_ok.sum().item()
            total_seqs += preds.size(0)
            # Token-level accuracy
            correct_tokens += ((preds == tgt_output) & mask).sum().item()
            total_tokens += mask.sum().item()
    print(
        f"\nTeacher-forcing seq accuracy: {correct_seqs}/{total_seqs} = {correct_seqs/total_seqs:.4f}"
    )
    print(
        f"Teacher-forcing token accuracy: {correct_tokens}/{total_tokens} = {correct_tokens/total_tokens:.4f}"
    )


def train_reinforce(model, dataloader, device):
    """Train the model using REINFORCE to maximize token-level reversal accuracy."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(RL_EPOCHS):
        total_reward = 0.0
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"RL Epoch {epoch+1}/{RL_EPOCHS}"):
            src = batch["src"].to(device)
            tgt = batch["tgt_output"].to(device)
            batch_size, seq_len = tgt.shape
            # Encoder outputs
            enc_mask = model._create_padding_mask(src)
            enc_out = model.encoder(src, enc_mask)
            # Initialize decoder input
            dec_in = torch.full(
                (batch_size, 1), BOS_TOKEN, dtype=torch.long, device=device
            )
            log_probs = torch.zeros(batch_size, device=device)
            preds = torch.zeros_like(tgt)
            # Autoregressive sampling
            for t in range(seq_len):
                mask = model._create_look_ahead_mask(dec_in.size(1), device)
                hid = model.decoder(dec_in, enc_out, mask, enc_mask)
                logits = model.final_linear(hid[:, -1, :])
                dist = Categorical(logits=logits)
                tok = dist.sample()
                log_probs += dist.log_prob(tok)
                preds[:, t] = tok
                dec_in = torch.cat([dec_in, tok.unsqueeze(1)], dim=1)
            # Compute token-level reward
            mask_tgt = tgt != PAD_TOKEN
            correct_tokens = ((preds == tgt) & mask_tgt).sum(dim=1).float()
            lengths = mask_tgt.sum(dim=1).float().clamp(min=1)
            rewards = correct_tokens / lengths
            # REINFORCE loss
            loss = -(rewards * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_reward += rewards.sum().item()
        avg_reward = total_reward / (len(dataloader.dataset))
        avg_loss = total_loss / len(dataloader)
        print(
            f"RL Epoch {epoch+1}/{RL_EPOCHS} - Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}"
        )


if __name__ == "__main__":
    print("\n--- Setting up Dataset & DataLoader ---")
    dataset = SequenceReversalDataset(
        num_samples=NUM_SAMPLES,
        min_len=MIN_SEQ_LEN,
        max_len=MAX_SEQ_LEN,
        min_num=MIN_NUM,
        max_num=MAX_NUM,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
    )

    # Use a simple collate_fn (already padded in dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Verify a batch
    print(f"Created DataLoader with {len(dataloader)} batches.")
    sample_batch = next(iter(dataloader))
    print("Sample batch keys:", sample_batch.keys())
    print("Sample src shape:", sample_batch["src"].shape)  # (batch_size, max_len + 1)
    print("Sample tgt_input shape:", sample_batch["tgt_input"].shape)
    print("Sample tgt_output shape:", sample_batch["tgt_output"].shape)

    print("\nSample source sequence (first 5):")
    print(sample_batch["src"][0, :15])
    print("Sample target input sequence (first 5):")
    print(sample_batch["tgt_input"][0, :15])
    print("Sample target output sequence (first 5):")
    print(sample_batch["tgt_output"][0, :15])

    # --- Initialize Model, Loss, Optimizer (Next Step) --- #
    print("\n--- Initializing Model, Loss, Optimizer ---")

    model = EncoderDecoder(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        input_vocab_size=VOCAB_SIZE,  # Same vocab for src and tgt
        target_vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN + 1,  # Account for BOS/EOS
        dropout_prob=DROPOUT_PROB,
        padding_idx=PAD_TOKEN,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized on {DEVICE}. Trainable parameters: {num_params:,}")
    print(f"Loss Function: {criterion}")
    print(f"Optimizer: {optimizer}")

    # --- Training Loop with Professor Forcing --- #
    print("\n--- Starting Training with Professor Forcing ---")
    model.train()
    consistency_criterion = nn.MSELoss()
    pf_lambda = 1.0  # Increased lambda for stronger consistency enforcement
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True
        )
        for batch in progress_bar:
            src = batch["src"].to(DEVICE)
            tgt_input = batch["tgt_input"].to(DEVICE)
            tgt_output = batch["tgt_output"].to(DEVICE)
            # Encode source
            enc_mask = model._create_padding_mask(src)
            encoder_output = model.encoder(src, enc_mask)
            # Teacher-forced run
            tgt_pad_mask = model._create_padding_mask(tgt_input)
            tgt_la_mask = model._create_look_ahead_mask(tgt_input.size(1), DEVICE)
            teacher_mask = torch.logical_or(tgt_pad_mask, tgt_la_mask)
            teacher_hidden = model.decoder(
                target=tgt_input,
                encoder_output=encoder_output,
                target_mask=teacher_mask,
                encoder_mask=enc_mask,
            )
            logits = model.final_linear(teacher_hidden)
            ce_loss = criterion(logits.view(-1, VOCAB_SIZE), tgt_output.view(-1))
            # Free-running run
            batch_size = src.size(0)
            seq_len = tgt_output.size(1)
            free_input = torch.full(
                (batch_size, 1), BOS_TOKEN, dtype=torch.long, device=DEVICE
            )
            free_hiddens = []
            for _ in range(seq_len):
                mask = model._create_look_ahead_mask(free_input.size(1), DEVICE)
                hid = model.decoder(free_input, encoder_output, mask, enc_mask)
                # Record hidden state
                free_hiddens.append(hid[:, -1, :])
                # Compute logits for prediction
                logits_free = model.final_linear(
                    hid[:, -1, :]
                )  # shape: (batch_size, vocab_size)
                next_tok = logits_free.argmax(dim=-1)
                free_input = torch.cat([free_input, next_tok.unsqueeze(1)], dim=1)
            free_hiddens = torch.stack(free_hiddens, dim=1)
            teacher_act = teacher_hidden
            consistency_loss = consistency_criterion(
                free_hiddens, teacher_act[:, : free_hiddens.size(1), :]
            )
            loss = ce_loss + pf_lambda * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} finished. Avg Loss: {avg_loss:.4f}")
    print("\n--- Training Complete ---")
    teacher_forcing_eval(model, dataloader, DEVICE)
    evaluate_model(model, num_eval_samples=10, device=DEVICE)
    print("\n--- Script finished ---")

    # Now experiment with REINFORCE training
    print("\n--- Starting REINFORCE Training Experiment ---")
    train_reinforce(model, dataloader, DEVICE)
    # Re-evaluate after RL
    print("\n--- Post-RL Evaluation ---")
    teacher_forcing_eval(model, dataloader, DEVICE)
    evaluate_model(model, num_eval_samples=10, device=DEVICE)
    print("\n--- RL Experiment Complete ---")

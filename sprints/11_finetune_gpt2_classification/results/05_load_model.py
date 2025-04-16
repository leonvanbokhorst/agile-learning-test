"""
Loads the pre-trained GPT-2 model with a classification head.
Sprint 11 - Task 3
"""

# pip install torch transformers

import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

# Configuration
MODEL_CHECKPOINT = "gpt2"
NUM_LABELS = 2  # Binary classification (Fake/Real)


def main():
    """Loads the model and prints its structure."""
    print(
        f"Loading pre-trained model: {MODEL_CHECKPOINT} with a classification head..."
    )

    try:
        # Load the model configuration first to potentially inspect/modify if needed
        # (Though we're just using defaults here)
        config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
        # Add label information to the config (needed by SequenceClassification model)
        config.num_labels = NUM_LABELS
        # If we use a tokenizer where pad != eos, we might need this:
        # config.pad_token_id = tokenizer.pad_token_id
        print(f"Model config loaded. Set num_labels to {config.num_labels}.")

        # Load the model with the specified configuration
        # This will download weights if not cached
        # It automatically adds a classification head for num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT,
            config=config,
            # Or more simply: num_labels=NUM_LABELS
        )
        print("Model loaded successfully!")

        # Print the model architecture
        print("\nModel Architecture:")
        print(model)

        # You can specifically look at the classifier layer:
        print("\nClassifier Layer Details:")
        # The actual attribute name might vary slightly, but often it's 'score' or 'classifier'
        if hasattr(model, "score"):
            print(model.score)
        elif hasattr(model, "classifier"):
            print(model.classifier)
        else:
            print("Could not find a specific 'score' or 'classifier' attribute.")

        print("\nNote the final Linear layer outputting num_labels (2) logits.")

    except Exception as e:
        print(f"\nAn error occurred while loading the model: {e}")
        print(
            "Ensure 'transformers' is installed, the checkpoint name is correct, and you have internet access for download."
        )

    print("\nScript finished.")


if __name__ == "__main__":
    main()

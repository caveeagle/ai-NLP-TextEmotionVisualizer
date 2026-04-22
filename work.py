
import torch

from transformers import RobertaTokenizer
from transformers import AutoModelForSequenceClassification

##########################################

file_path = 'relevant_text.txt'

model_name = 'j-hartmann/emotion-english-distilroberta-base'

DEBUG = 1

##########################################

with open(file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

tokenizer = RobertaTokenizer.from_pretrained(model_name)

# ====================================================
# Tokenization using Byte-Pair Encoding (BPE)
# ====================================================
tokens = tokenizer.tokenize(raw_text)

if(DEBUG):
    print(f'\nTotal number of tokens: {len(tokens)}\n')

# ====================================================
# STEP 2. Add special markers <s> and </s>
# ====================================================
bos_token = tokenizer.bos_token  # <s>   — beginning of sequence
eos_token = tokenizer.eos_token  # </s>  — end of sequence

tokens_with_special = [bos_token] + tokens + [eos_token]

if(DEBUG):
    print('=' * 60)
    print('STEP 2 — SPECIAL TOKENS')
    print('=' * 60)
    print(f'Beginning-of-sequence token: {bos_token}')
    print(f'End-of-sequence token:       {eos_token}')
    print(f'Tokens after adding markers (first 5):  {tokens_with_special[:5]}')
    print(f'Tokens after adding markers (last 5):   {tokens_with_special[-5:]}')
    print(f'Total token count (with markers): {len(tokens_with_special)}')
    print()

# ====================================================
# STEP 3. Convert tokens to numeric IDs
# ====================================================
token_ids = tokenizer.convert_tokens_to_ids(tokens_with_special)

# ====================================================
# STEP 4. Padding / Truncation to fixed length
# ====================================================
MAX_LENGTH = 512
pad_token_id = tokenizer.pad_token_id  # PAD token ID (usually 1 for RoBERTa)

if len(token_ids) > MAX_LENGTH:
    # Truncation: keep first (MAX_LENGTH-1) tokens + closing </s>
    input_ids = token_ids[:MAX_LENGTH - 1] + [tokenizer.eos_token_id]
    num_truncated = len(token_ids) - MAX_LENGTH
    num_padded = 0
else:
    # Padding: fill up to MAX_LENGTH with PAD tokens
    num_padded = MAX_LENGTH - len(token_ids)
    num_truncated = 0
    input_ids = token_ids + [pad_token_id] * num_padded

# ====================================================
# STEP 5. Create the attention mask
# ====================================================
# 1 — real token, 0 — PAD filler
real_token_count = len(token_ids) if len(token_ids) <= MAX_LENGTH else MAX_LENGTH
attention_mask = [1] * real_token_count + [0] * num_padded

# ====================================================
# RESULT: two arrays ready to be fed into the model
# ====================================================

if(DEBUG):
    print('=' * 60)
    print('RESULT')
    print('=' * 60)
    print(f'input_ids      — length {len(input_ids)}')
    print(f'attention_mask  — length {len(attention_mask)}')
    print()
    print('Both arrays are ready to be passed into the RoBERTa model.')

######################################################
# ====================================================
# STAGE 3. MODEL INFERENCE (emotion prediction)
# ====================================================
######################################################


# ----------------------------------------------------
# STEP 1. Feed data into the model
# ----------------------------------------------------
# Load a pre-trained RoBERTa model fine-tuned for emotion classification

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # switch to inference mode (disable dropout etc.)

# Convert lists to PyTorch tensors and add batch dimension
input_ids_tensor = torch.tensor([input_ids])            # shape: [1, 512]
attention_mask_tensor = torch.tensor([attention_mask])   # shape: [1, 512]

# ----------------------------------------------------
# STEP 2-3. Forward pass through transformer layers
#           and classification head
# ----------------------------------------------------
# No gradient computation needed during inference

with torch.no_grad():
    outputs = model(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor
    )

# Raw scores (logits) — one per emotion class
logits = outputs.logits  # shape: [1, num_emotions]

# ----------------------------------------------------
# STEP 4. Convert logits to probabilities (softmax)
# ----------------------------------------------------
probabilities = torch.softmax(logits, dim=-1)[0]  # shape: [num_emotions]

assert round(probabilities.sum().item(), 4) == 1.0, 'Probabilities do not sum to 1.0'

# Get emotion labels from the model config
id2label = model.config.id2label  # e.g. {0: 'anger', 1: 'disgust', ...}

# ----------------------------------------------------
# STEP 5. Sort and select top emotions
# ----------------------------------------------------
sorted_indices = torch.argsort(probabilities, descending=True)

print('=' * 60)
print('  EMOTIONS  ')
print('=' * 60)

for rank, idx in enumerate(sorted_indices, start=1):
    label = id2label[idx.item()]
    prob = probabilities[idx].item()
    marker = ' <-- TOP' if rank <= 2 else ''
    print(f'  {rank}. {label:<12s}  {prob:.4f}  ({prob * 100:.1f}%){marker}')

# Save top-1 and top-2 for future color mapping
top1_idx = sorted_indices[0].item()
top2_idx = sorted_indices[1].item()
top1_emotion = id2label[top1_idx]
top2_emotion = id2label[top2_idx]
top1_prob = probabilities[top1_idx].item()
top2_prob = probabilities[top2_idx].item()

print()
print(f'Primary emotion:   {top1_emotion} ({top1_prob * 100:.1f}%)')
print(f'Secondary emotion: {top2_emotion} ({top2_prob * 100:.1f}%)')
print()

print(f'Job finished')


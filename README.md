# Transformer-based Text Summarization Project

## Project Overview
This project aimed to develop a **text summarizer** using a Transformer-based model. The task was to convert longer texts (dialogues) into **concise summaries**, extracting key information for easier understanding, similar to summarizing a movie plot or a research abstract.

---

## Tools and Technologies Used
- **Hugging Face**  
  - `transformers` library: Access and work with Transformer models.  
  - `datasets` library: Load and manage datasets.  
  - `evaluate` library: Calculate evaluation metrics like ROUGE.  
  - Key Components:
    - `AutoTokenizer`: Initialize tokenizers for chosen models.  
    - `AutoModelForSeq2SeqLM`: Load sequence-to-sequence models for summarization.  
    - `pipeline`: High-level API for NLP tasks and inference.  
    - `Trainer` & `TrainingArguments`: Streamline training and hyperparameter management.  
    - `DataCollatorForSeq2Seq`: Efficient batching and token padding.  

- **PyTorch**: Deep learning framework for model training.  
- **Google Colab**: Development environment with GPU support (e.g., Tesla T4) and high RAM.

---

## Dataset
**Samsung Dialogue Dataset**  
- ~16,000 conversational texts from messenger/WhatsApp-like interactions.  
- **Columns:**  
  - `ID`: Unique identifier  
  - `dialogues`: Raw conversational text (input)  
  - `summary`: Expert-written summary (target/label)  
- **Splits:**  
  - Training: 14,732 examples  
  - Validation: 818 examples  
  - Test: 819 examples  
- **Significance:** Pre-labeled dataset reduces manual annotation effort.

---

## Model Selection
- **Pre-trained Model:** Google Pegasus CNN Daily Mail (Transformer-based seq2seq model).  
- **Evaluation of Pre-trained Model:**  
  - Tested on Samsung dataset before fine-tuning.  
  - ROUGE score was very low (~0.01), indicating the need for fine-tuning.

---

## Methodology

### 1. Data Preprocessing & Tokenization
- Used `AutoTokenizer` to tokenize dialogues and summaries.  
- Converted text into `input_ids`, `attention_mask`, and `labels`.  
- Max token length set to 1024.  
- Applied `DataCollatorForSeq2Seq` for dynamic padding and batching.

### 2. Fine-tuning (Transfer Learning)
- **Strategy:** Adapt pre-trained Pegasus to custom dataset.  
- **TrainingArguments:** Defined for output directory, epochs, batch size, weight decay, logging, and evaluation.  
- **Trainer:** Integrated model, tokenizer, data collator, and datasets.  
- **Training:** Fine-tuned the model (demo with 1 epoch; recommended 100–200 epochs in practice).

### 3. Post-Fine-tuning Evaluation & Inference
- Re-calculated ROUGE score to evaluate improvements.  
- Saved fine-tuned model and tokenizer for later use.  
- Used `pipeline` for inference on new dialogues.  
- Controlled summary length using `length_penalty`.  
- Fine-tuned model produced high-quality summaries compared to human references.

---

## Key Highlights
- Built a **Transformer-based seq2seq summarizer** using PyTorch and Hugging Face.  
- Leveraged **transfer learning** to adapt Pegasus to a domain-specific dataset.  
- Incorporated **custom tokenization and data collators** for efficient training.  
- Evaluated using **ROUGE metric** and demonstrated significant improvement after fine-tuning.  
- Modular codebase allows easy adaptation to other datasets or language pairs.  

---

## Usage
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("path_to_finetuned_model")
model = AutoModelForSeq2SeqLM.from_pretrained("path_to_finetuned_model")

# Initialize summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Example dialogue
dialogue_text = "Your dialogue text here..."

# Generate summary
summary = summarizer(dialogue_text, max_length=150, min_length=30, length_penalty=2.0)
print(summary)

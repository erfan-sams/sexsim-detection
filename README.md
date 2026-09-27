# Sexism Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Can a model tell whether a social-media post is sexist?** This repo tries two ways to find out:

1. **Train a classifier:** a simple BiLSTM vs. a fine-tuned RoBERTa.
2. **Prompt an LLM:** ask Phi-3 and Mistral directly, with and without examples.

Both parts were assignments for the NLP course in the Master's in Artificial Intelligence at the University of Bologna.

> [!WARNING]
> The datasets and notebook outputs contain sexist and offensive language.

## Results at a glance

| Part | Best setup | Accuracy |
|---|---|---|
| 1. Train a classifier | Fine-tuned RoBERTa | **0.83** |
| 2. Prompt an LLM | Mistral-7B with 4 examples | **0.75** |

The two parts use different datasets, so compare models within a part, not across parts.

**Key takeaways**

- **Fine-tuning wins, but not by much.** A simple BiLSTM with GloVe word vectors reaches 0.78 F1, against 0.83 for RoBERTa.
- **Examples in the prompt can matter a lot.** Mistral went from 0.59 to 0.75 accuracy once it saw 4 labeled examples. Phi-3 didn't improve.
- **Every model has the same blind spots.** Rude text with nothing to do with gender gets flagged as sexist. Subtle sexism phrased in friendly language slips through.

## What's in this repo

| File | What it is | Open |
|---|---|---|
| [`transformer-sexsim-detection.ipynb`](transformer-sexsim-detection.ipynb) | Part 1 notebook: BiLSTM and RoBERTa | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erfan-sams/sexsim-detection/blob/main/transformer-sexsim-detection.ipynb) |
| [`report-transformer.pdf`](report-transformer.pdf) | Part 1 report | |
| [`LLMs_sexism_detection.ipynb`](LLMs_sexism_detection.ipynb) | Part 2 notebook: prompting LLMs | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erfan-sams/sexsim-detection/blob/main/LLMs_sexism_detection.ipynb) |
| [`report-llms.pdf`](report-llms.pdf) | Part 2 report | |

---

## Part 1: Train a classifier

**The question:** Does a large pre-trained transformer beat a simple recurrent network on this task?

**The data:** English tweets from [EXIST 2023](https://clef2023.clef-initiative.eu/index.php?page=Pages/labs.html#EXIST) (Task 1). Six people labeled each tweet, and the majority vote is the final label; tweets with a 3–3 tie were dropped. That leaves 2,870 tweets for training, 158 for validation and 286 for testing.

**The models:**

- **BiLSTM:** one or two bidirectional LSTM layers on top of frozen GloVe word vectors.
- **RoBERTa:** [`cardiffnlp/twitter-roberta-base-hate`](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate), a model already trained on hate speech, fine-tuned on this data.

**Results** (286 test tweets):

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| BiLSTM, 1 layer | 0.77 ± 0.01 | 0.77 |
| BiLSTM, 2 layers | 0.76 ± 0.02 | 0.78 |
| **RoBERTa, fine-tuned** | **0.83** | **0.83** |

*Each BiLSTM was trained with 3 random seeds. Its accuracy is the mean ± std across the seeds, and its F1 comes from the seed with the best test accuracy. RoBERTa was trained once.*

**What stood out:**

- RoBERTa catches more of the sexist tweets: its recall on the sexist class is 0.87, against 0.70 for the best BiLSTM.
- A second LSTM layer made no real difference.
- Both models were often fooled by surface cues. Rude tweets that don't mention gender were flagged as sexist, and neutral tweets that mention a gender were sometimes flagged too.

**Download the fine-tuned RoBERTa weights:** [Google Drive](https://drive.google.com/drive/folders/1xYZe0XRxP0o6gbpyFjruP68642eZbNaC?usp=sharing)

<details>
<summary>Training details</summary>

- **Text cleaning (BiLSTM):** removed URLs, mentions, hashtags and non-letter characters, then lemmatized with spaCy.
- **BiLSTM:** GloVe 6B 300-d (frozen), 128 LSTM units, sigmoid output, Adam (lr 1e-3), batch size 64, 6 epochs. The epoch with the best validation F1 was kept. Seeds: 42, 60, 1337.
- **RoBERTa:** Hugging Face `Trainer` with lr 2e-5, batch size 16, 3 epochs. The epoch with the best validation F1 was kept.

</details>

---

## Part 2: Prompt an LLM

**The question:** Can an LLM spot sexism with no training at all, just a prompt? And does showing it a few examples help?

**The data:** 300 posts from [EDOS](https://github.com/rewire-online/edos) (Task A), half sexist and half not. The few-shot examples come from a separate file of 1,000 labeled posts.

**The models:** Both run in 4-bit precision so they fit on a single GPU.

- [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (`microsoft/Phi-3-mini-4k-instruct`)
- [Mistral-7B v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) (`mistralai/Mistral-7B-Instruct-v0.3`)

**How it works:** The model reads a post and must answer only `YES` (sexist) or `NO`.

- **Zero-shot:** just the post.
- **Few-shot:** the post plus 4 labeled examples (2 sexist, 2 not).

**Results** (accuracy on 300 posts):

| Model | Zero-shot | Few-shot |
|---|---|---|
| Phi-3-mini | 0.64 | 0.64 |
| **Mistral-7B** | 0.59 | **0.75** |

*Macro F1 follows the same pattern: Phi-3 goes from 0.64 to 0.63, Mistral from 0.52 to 0.75. The zero-shot numbers come from the notebook; the report only covers few-shot.*

**What stood out:**

- Without examples, Mistral labeled most posts as sexist, including 79% of the ones that weren't. With examples, it got much more balanced.
- Phi-3-mini did not benefit from the examples.
- Same blind spots as Part 1. Stereotypes phrased in a friendly way were missed, and crude but non-sexist posts were flagged.

<details>
<summary>Prompt and evaluation details</summary>

The prompt given by the assignment:

```text
System: You are an annotator for sexism detection.
User:   Your task is to classify input text as containing sexism or not. Respond only YES or NO.

        TEXT:
        {text}

        ANSWER:
```

- The few-shot version adds an `EXAMPLES:` block before `TEXT:`.
- Decoding is greedy (no sampling).
- The few-shot examples are sampled at random without a fixed seed, so a rerun can give different numbers.
- Any answer other than a clean `YES` or `NO` counts as "not sexist". This happened with 5% of Mistral's few-shot answers.

</details>

---

## Run it yourself

The notebooks were built for Google Colab, so the simplest route is the **Open in Colab** buttons above.

**1. Get the data.** The data isn't included in this repo. Download it from the course repository [lt-nlp-lab-unibo/nlp-course-material](https://github.com/lt-nlp-lab-unibo/nlp-course-material).

| Part | Folder | Files | What to do |
|---|---|---|---|
| 1 | [`2024-2025/Assignment 1/data`](https://github.com/lt-nlp-lab-unibo/nlp-course-material/tree/main/2024-2025/Assignment%201/data) | `training.json`, `validation.json`, `test.json` | Change the file paths in the notebook, which point to a Google Drive folder |
| 2 | `Assignment 2/data` | `a2_test.csv`, `demonstrations.csv` | Put them next to the notebook |

**2. Check the requirements.**

- **Part 1:** TensorFlow, `transformers`, `datasets`, spaCy with `en_core_web_sm`, gensim and scikit-learn. The notebook downloads GloVe on its own.
- **Part 2:** a CUDA GPU and `transformers`, `accelerate` and `bitsandbytes`. You also need a Hugging Face account: accept Mistral's terms on its model page if asked, then log in with `huggingface-cli login`.

## Credits

- **Author:** Erfan Samieyan Sahneh, Master's in Artificial Intelligence, University of Bologna
- **Assignments designed by:** Federico Ruggeri, Eleonora Mancini and Paolo Torroni (NLP course)
- **License:** [MIT](LICENSE)

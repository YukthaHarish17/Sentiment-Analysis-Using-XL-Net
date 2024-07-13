# Sentiment-Analysis-using-XL-Net

This repository contains code for sentiment analysis using the XLNet model. The dataset used is the IMDB movie reviews dataset.

## Setup

To run the code, you need to have the following Python packages installed:

- `watermark`
- `torch`
- `transformers`
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `sentencepiece`
- `keras`
- `tensorflow`

You can install these packages using pip:

```sh
pip install -q -U watermark
pip install torch transformers numpy pandas seaborn matplotlib scikit-learn sentencepiece keras tensorflow
```

## Loading Data

The dataset used is the IMDB movie reviews dataset. You can download it from [here](https://ai.stanford.edu/~amaas/data/sentiment/). Place the `IMDB Dataset.csv` file in the appropriate directory.

## Data Preprocessing

1. **Shuffle the Dataset**:
    ```python
    from sklearn.utils import shuffle
    df = shuffle(df)
    df.head()
    ```

2. **Clean the Text**:
    ```python
    import re
    def clean_text(text):
        text = re.sub(r"@[A-Za-z0-9]+", '', text)
        text = re.sub(r"https?://[A-Za-z0-9./]+",'',text)
        text = re.sub(r"[^a-zA-Z.!?'0-9]", '', text)
        text = re.sub('/t','',text)
        text = re.sub(r" +", '', text)
        return text
    df['review'] = df['review'].apply(clean_text)
    ```

3. **Sentiment Label Encoding**:
    ```python
    def sentiment2label(sentiment):
        if sentiment == "positive":
            return 1
        else:
            return 0
    df['sentiment'] = df['sentiment'].apply(sentiment2label)
    ```

## Data Visualization

To visualize the sentiment distribution:

```python
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 8, 6
sns.countplot(df.sentiment)
plt.xlabel('review score')
```

## Tokenization with XLNet

1. **Install SentencePiece**:
    ```sh
    pip install sentencepiece
    ```

2. **Tokenize the Input Text**:
    ```python
    from transformers import XLNetTokenizer, XLNetModel
    PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
    tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    input_txt = "India is my country. All Indians are my brothers and sisters"
    encodings = tokenizer.encode_plus(input_txt, add_special_tokens=True, max_length=16, return_tensors='pt', truncation=True)
    print('input_ids : ', encodings['input_ids'])
    tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])
    ```

3. **Attention Mask**:
    ```python
    from keras.preprocessing.sequence import pad_sequences
    attention_mask = pad_sequences(encodings['attention_mask'], maxlen=512, dtype=torch.Tensor, truncating="post", padding="post")
    attention_mask = attention_mask.astype(dtype='int64')
    attention_mask = torch.tensor(attention_mask)
    attention_mask.flatten()
    ```

## Token Length Distribution

To visualize the token length distribution:

```python
token_lens = []
for txt in df['review']:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 1024])
plt.xlabel('Token count')
```

## Device Configuration

The code runs on the CPU by default, but if you have a GPU, it can be configured to use it:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
```

## Results

### Dataset Overview

Here is a sample of the dataset after shuffling and cleaning:

| Review                                                                                                                                       | Sentiment |
|----------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| This is definitely a "must see" film. The exce...                                                                                             | positive  |
| It's a bit difficult to believe that this came...                                                                                             | negative  |
| Diane Lane is beautiful and sexy, and Tuscany ...                                                                                            | negative  |
| I recently stumbled across a TV showing of "Pa...                                                                                            | positive  |
| This is a stupid movie. Like a lot of these ka...                                                                                            | negative  |

### Sentiment Distribution

The dataset contains a balanced distribution of positive and negative reviews:

```plaintext
1    12100
0    11900
Name: sentiment, dtype: int64
```

### Tokenization Example

Input text: "India is my country. All Indians are my brothers and sisters"

Tokenized output:

```plaintext
['▁India', '▁is', '▁my', '▁country', '.', '▁All', '▁Indians', '▁are', '▁my', '▁brothers', '▁and', '▁sisters', '<sep>', '<cls>']
```

### Attention Mask

Attention mask for the tokenized input:

```plaintext
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0])
Here's a README file for the provided code, including explanations and expected outputs.

---

# Token Length Distribution Analysis

This repository contains code to analyze the distribution of token lengths in text data using the Hugging Face tokenizer. The code reads reviews from a DataFrame, tokenizes the text, and plots the distribution of token lengths.

## Requirements

- Python 3.6+
- Transformers (Hugging Face)
- Pandas
- Seaborn
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/token-length-distribution.git
    cd token-length-distribution
    ```

2. Install the required packages:
    ```bash
    pip install transformers pandas seaborn matplotlib
    ```

## Usage

1. Ensure your data is in a DataFrame format with a column named `review`.

2. Run the following code to analyze the token length distribution:

    ```python
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    # Load your DataFrame here
    # Example:
    # df = pd.read_csv('your_data.csv')

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Calculate token lengths
    token_lens = []
    for txt in df['review']:
        tokens = tokenizer.encode(txt, max_length=512)
        token_lens.append(len(tokens))

    # Plot the distribution
    sns.histplot(token_lens, kde=True)
    plt.xlim([0, 1024])
    plt.xlabel('Token count')
    plt.title('Token Length Distribution')
    plt.show()
    ```

## Explanation

1. **Tokenizer Initialization**:
    ```python
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ```
    This initializes the BERT tokenizer from Hugging Face's transformers library.

2. **Token Length Calculation**:
    ```python
    token_lens = []
    for txt in df['review']:
        tokens = tokenizer.encode(txt, max_length=512)
        token_lens.append(len(tokens))
    ```
    This loop iterates over each review in the DataFrame, tokenizes the text, and appends the length of the tokens to the `token_lens` list.

3. **Distribution Plotting**:
    ```python
    sns.histplot(token_lens, kde=True)
    plt.xlim([0, 1024])
    plt.xlabel('Token count')
    plt.title('Token Length Distribution')
    plt.show()
    ```
    This plots the distribution of token lengths using Seaborn's `histplot` function. The plot includes a kernel density estimate (KDE) to show the density of token lengths.

## Outputs

- **Token Length Distribution Plot**: A histogram showing the distribution of token lengths with a KDE overlay.
  ![Token Length Distribution](path_to_your_plot_image)

## Notes

- The code includes a warning regarding the use of `distplot`, which is deprecated. The code has been updated to use `histplot`.
- Adjust the `max_length` parameter as needed for your specific tokenizer and text data.

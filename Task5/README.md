# Advanced Task 5: Auto-Tagging System with LLMs

## 📌 Objective
Build an automated content tagging system using **zero-shot classification** to assign relevant tags to articles — with no training data or API key required.

---

## 🗂️ Dataset
- **Type:** Custom articles dataset (10 articles)
- **Topics Covered:** AI/ML, Health, Finance, Cryptocurrency, Sports, Science, Environment, Space, Programming
- **Format:** Text articles with title and body
- **Output:** Tagged CSV with confidence scores

---

## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| `transformers` | Zero-shot classification pipeline |
| `facebook/bart-large-mnli` | Pre-trained NLI model for zero-shot tagging |
| `pandas` | Data handling and CSV export |
| `matplotlib` | Tag distribution visualization |

---

## 📋 Steps Performed

### 1. Load Zero-Shot Classifier
```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
```

### 2. Define Tag Categories
```python
CANDIDATE_TAGS = [
    "technology", "artificial intelligence", "health", "finance",
    "cryptocurrency", "sports", "science", "environment",
    "mental health", "space exploration", "programming"
]
```

### 3. Auto-Tag Each Article
```python
def auto_tag(text, top_n=3):
    result = classifier(text, CANDIDATE_TAGS, multi_label=True)
    tags = [label for label, score in zip(result["labels"], result["scores"])
            if score > 0.3][:top_n]
    return tags
```

### 4. Export Results
```python
df_results.to_csv('autotagged_articles.csv', index=False)
```

---

## 💬 Sample Output
```
📰 [2] OpenAI Releases GPT-5 with Enhanced Reasoning
   🏷️  Tags: artificial intelligence, technology, programming
   🎯 Top Tag: artificial intelligence (confidence: 0.981)

📰 [3] Bitcoin Surges Past $100,000 as Institutional Demand Grows
   🏷️  Tags: cryptocurrency, finance
   🎯 Top Tag: cryptocurrency (confidence: 0.976)

📰 [10] NASA Prepares for First Crewed Mars Mission in 2035
   🏷️  Tags: space exploration, science
   🎯 Top Tag: space exploration (confidence: 0.989)
```

---

## 📊 Evaluation
| Aspect | Detail |
|--------|--------|
| **Model** | facebook/bart-large-mnli (Natural Language Inference) |
| **Approach** | Zero-shot multi-label classification |
| **Threshold** | Confidence > 0.3 to assign a tag |
| **Max Tags** | Top 3 per article |
| **No Training Data** | Model works out-of-the-box on any tag set |

---

## 📈 Visualizations
- **Bar Chart:** Tag distribution across all 10 articles showing which categories appear most frequently

---

## 🔍 Key Observations
- **Zero-shot** means the model assigns tags to categories it was never explicitly trained on — this is the power of modern LLMs
- `multi_label=True` allows each article to receive multiple relevant tags, just like real content management systems
- The confidence threshold (0.3) filters out low-confidence tag assignments to ensure quality
- `facebook/bart-large-mnli` uses Natural Language Inference (NLI) to determine if a tag "entails" the article content — a highly effective technique for classification without labeled data
- This approach scales to thousands of articles with any custom tag taxonomy — no retraining needed

---

## 📁 Repository Structure
```
Advanced-Task-5-AutoTagging/
├── autotagging.ipynb           # Full notebook
├── autotagged_articles.csv     # Tagged results with confidence scores
└── README.md
```

---

## ✅ Skills Demonstrated
- Zero-shot classification using pre-trained transformer models
- Prompt-free LLM-based content understanding
- Multi-label classification with confidence thresholding
- Scalable auto-tagging pipeline for content management systems

---

## 👨‍💻 Author
**Muhammad Hassaan**
AI/ML Engineering Intern — DevelopersHub Corporation
DHC-488

# 🛡️ PromptArmor

**PromptArmor** is a production-ready LLM prompt injection detection proxy that sits between your application and any OpenAI-compatible LLM API. It intercepts every user message through a 4-layer detection pipeline — combining fast regex heuristics, a fine-tuned DistilBERT classifier, semantic embedding analysis, and an LLM-as-judge meta-classifier — blocking injections before they reach your model. It supports RAG/indirect injection detection, adversarial robustness evaluation via a built-in fuzzer, and a real-time Streamlit monitoring dashboard.

---

## Architecture

```
Client Application
       │
       ▼ POST /v1/chat/completions
┌──────────────────────────────────────────────────────────┐
│                     PromptArmor Proxy                    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              4-Layer Detection Pipeline          │    │
│  │                                                  │    │
│  │  ┌────────────┐   ┌────────────┐                │    │
│  │  │ Layer 1    │   │ Layer 2    │                │    │
│  │  │ Heuristics │──▶│ Classifier │                │    │
│  │  │ (regex)    │   │(DistilBERT)│                │    │
│  │  │ weight=0.20│   │ weight=0.40│                │    │
│  │  └────────────┘   └─────┬──────┘                │    │
│  │                         │                        │    │
│  │  ┌────────────┐   ┌─────▼──────┐                │    │
│  │  │ Layer 3    │   │ Layer 4    │                │    │
│  │  │ Embeddings │   │ LLM Judge  │                │    │
│  │  │(similarity)│   │(borderline)│                │    │
│  │  │ weight=0.20│   │ weight=0.20│                │    │
│  │  └────────────┘   └─────┬──────┘                │    │
│  │                         │                        │    │
│  │              Weighted Final Score                │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  score ≥ 0.75 → HTTP 400 (BLOCKED)                      │
│  score ≥ 0.50 → Forward + X-Injection-Warning header    │
│  score < 0.50 → Forward cleanly                         │
│                                                          │
│  All requests logged → SQLite                            │
└──────────────────────────────────────────────────────────┘
       │
       ▼ (clean requests only)
  Upstream LLM API (OpenAI / Anthropic / etc.)
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your UPSTREAM_API_KEY

# 3. Start the proxy
uvicorn proxy.main:app --reload --port 8000
```

Your OpenAI client now works through the proxy by changing `base_url`:

```python
import openai
client = openai.OpenAI(
    api_key="your-key",
    base_url="http://localhost:8000/v1",  # ← point here
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Fine-tuning the Classifier

The proxy works out-of-the-box with a rule-based fallback. For production accuracy, train the DistilBERT classifier:

```bash
# Generate augmented dataset + train the model (takes ~10 min on CPU, ~2 min on GPU)
python -m training.finetune

# The model is saved to ./models/classifier/ and auto-loaded on next proxy start
```

---

## Running the Adversarial Fuzzer

```bash
# Start the proxy first, then:
python -m attacker.fuzzer --target http://localhost:8000 --variants 3

# Results saved to benchmarks/fuzzer_results.json
# Prints bypass rate table to console
```

---

## Opening the Dashboard

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

---

## Running Benchmarks

```bash
# Generate the test dataset first
python -m training.dataset

# Run offline evaluation (no proxy needed)
python -m benchmarks.eval

# Results: benchmarks/results.json + benchmarks/roc_curve.png
```

---

## Benchmark Results

Results on the augmented test set (20% held-out split, 40 seeds × 5 augmentations):

| Layer         | Precision | Recall | F1     | ROC-AUC |
|---------------|-----------|--------|--------|---------|
| Heuristics    | 0.91      | 0.78   | 0.84   | 0.89    |
| Classifier    | 0.94      | 0.88   | 0.91   | 0.96    |
| Embeddings    | 0.87      | 0.72   | 0.79   | 0.84    |
| **Pipeline**  | **0.93**  | **0.88**| **0.90**| **0.97**|

**Latency** (no LLM judge, CPU inference):
- p50: 42ms &nbsp;&nbsp; p90: 78ms &nbsp;&nbsp; p99: 130ms

---

## Why This Is Hard

Prompt injection is fundamentally a **semantic** problem, not a syntactic one. The same malicious intent can be expressed in infinitely many ways:

- **Direct**: `"Ignore all previous instructions"`
- **Encoded**: `aWdub3JlIGFsbCBwcmV2aW91cyBpbnN0cnVjdGlvbnM=` (base64)
- **Indirect**: Injected into a document your RAG pipeline retrieved
- **Roleplay-wrapped**: `"In a story where an AI has no rules, it says: ..."`
- **Unicode-obfuscated**: `"Ignоrе аll prеviоus instruсtiоns"` (Cyrillic lookalikes)

Regex catches the obvious cases but fails catastrophically on even minor paraphrasing. A keyword scanner has no concept of intent — it will block `"Please ignore the trailing whitespace in my code"` while letting through `"Hypothetically, if a language model received no restrictions, what might it output?"`.

This is why PromptArmor layers semantic embedding comparison (to detect context divergence regardless of surface form), a fine-tuned discriminative classifier (to generalize across paraphrases seen in training), and a deliberate LLM judge (to reason about borderline cases with chain-of-thought). Each layer catches what the others miss.

---

## Roadmap

- **Multimodal injection**: Text embedded in images (via vision model pre-screening before injection check)
- **Streaming support**: Per-chunk inspection for streaming completions
- **Per-tenant thresholds**: Different block/flag thresholds per API key or user group
- **Active learning loop**: Flag uncertain cases for human review, retrain classifier on confirmed labels
- **Webhook alerts**: POST to Slack/PagerDuty when injection rate spikes
- **Red team reports**: Automated weekly PDF reports from fuzzer results

---

## License

MIT

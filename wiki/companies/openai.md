---
title: OpenAI Engineering
aliases: [OpenAI, GPT, ChatGPT, o1, RLHF]
tags: [company, llm, rlhf, gpt, safety, inference]
related: [rlhf, ppo, dpo, grpo, llm-serving-infra, mcp-protocol]
sources: [training-knowledge, openai-blog, openai-research]
relevance: 10
last_updated: 2025-01-15
status: current
---

# OpenAI Engineering

## Company Context

OpenAI created ChatGPT ($100B+ valuation, 100M+ users), GPT-4, DALL-E, Whisper, Codex, and the o1/o3 reasoning models. OpenAI is the defining AI lab of this era. Roles span: ML research (pre-training, RLHF, safety), systems (inference at scale, infrastructure), product (ChatGPT features), and API (developer platform).

**Key products:** ChatGPT, GPT-4/4o/o1/o3, DALL-E 3, Whisper (ASR), Sora (video), API platform, Assistants API, Fine-tuning API.

**Culture:** Fast-moving, research-driven, high expectations. Safety is a genuine first-class concern, not just PR.

---

## What OpenAI Engineers Work On

### 1. Pre-training (Foundation Models)

Training GPT-4, o1, etc. at scale. Key engineering challenges:

```
GPT-4 training (estimated):
- ~1.8T parameters (MoE architecture, unconfirmed)
- ~25K A100 GPUs for ~90-100 days
- Training data: ~13T tokens
- Infrastructure: Azure (exclusive partnership)

Training stack:
- Megatron-LM for tensor + pipeline parallelism
- DeepSpeed for ZeRO optimization
- Custom CUDA kernels for attention, MoE routing
- Gradient checkpointing for memory efficiency
- BF16 mixed precision throughout
```

**Data pipeline at scale:**
```python
# Pre-training data requirements:
# 1. Web crawl: Common Crawl deduplication, quality filtering
# 2. Books: copyright-free + licensed
# 3. Code: GitHub (multiple programming languages)
# 4. Scientific papers: arXiv, PubMed
# 5. Curated high-quality sources

# Key challenge: deduplication at scale
# MinHash LSH for near-duplicate detection across trillions of tokens
from datasketch import MinHash, MinHashLSH

# Quality filtering:
# - Perplexity-based filtering (discard high-perplexity text)
# - FastText classifiers for quality/topic
# - Rule-based filters (word count, language, ratio of special chars)
```

### 2. RLHF Pipeline (Core Differentiator)

OpenAI's RLHF pipeline is what makes ChatGPT vs GPT-3 Davinci:

```
Stage 1: SFT (Supervised Fine-Tuning)
  - Human contractors write ~10K high-quality prompt-response pairs
  - Fine-tune GPT-4 base on these pairs
  - Creates helpful, instruction-following baseline

Stage 2: Reward Model Training
  - For each prompt, generate 4-9 responses (from SFT model)
  - Human contractors rank responses
  - Train reward model: Bradley-Terry loss on preference pairs
  - RM outputs scalar "helpfulness" score

Stage 3: PPO (Reinforcement Learning from Human Feedback)
  - Policy (SFT model) generates responses
  - Reward model scores them
  - PPO updates policy to maximize reward
  - KL penalty from SFT model prevents reward hacking
  - KL coefficient β ≈ 0.1-0.2

KL-penalized reward:
R(s,a) = r_RM(s,a) - β * KL(π_θ(·|s) || π_SFT(·|s))
```

### 3. o1 / o3 (Chain-of-Thought Reasoning)

OpenAI's o1 introduced "thinking before answering" via extended chain-of-thought:

```
o1 key insight:
- Trained to generate internal reasoning traces (hidden from user)
- Reasoning traces are longer = more "thinking time" = better accuracy
- Scaling test-time compute (not just training compute)
- MCTS or beam search over reasoning steps (unconfirmed)

Result: o1 vs GPT-4 on competition math:
- GPT-4o: ~13% on AIME 2024
- o1: ~83% on AIME 2024
```

**Engineering challenge:** Serving reasoning models is expensive — each response requires generating 1000s of "thinking" tokens before the actual answer.

### 4. ChatGPT Infrastructure

Serving 100M+ users with complex conversations:

```
ChatGPT request flow:
1. User message → moderation (content policy classifier)
2. Context assembly: conversation history + memory (if enabled) + tool results
3. Token count check: truncate if over context limit
4. Inference: GPT-4 (or mini) via Azure OpenAI
5. Streaming: Server-Sent Events → token-by-token to browser
6. Safety check: output moderation before display
7. Logging: usage tracking, RLHF data collection (with consent)

Capacity:
- Millions of concurrent conversations
- Tokens/second: O(100M) at peak
- Latency target: first token < 500ms, subsequent tokens < 50ms
```

### 5. Function Calling / Tool Use / Agents

```python
# OpenAI function calling: structured JSON tool use
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=tools,
    tool_choice="auto"
)

# If tool call: parse, execute, pass result back
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    result = call_weather_api(tool_call.function.arguments)
    # Continue conversation with tool result
```

---

## Key Questions

**Systems:**
- "How does ChatGPT serve 100M users — design the inference infrastructure"
- "Design the RLHF data collection and training pipeline"
- "How would you implement streaming responses for ChatGPT?"
- "Design a content moderation system for an LLM API"
- "How do you handle context length limits in a multi-turn conversation?"

**ML Research Depth:**
- "Explain the PPO algorithm for RLHF. Why is the KL penalty needed?"
- "What are the failure modes of reward models? What is reward hacking?"
- "How does DPO compare to PPO for alignment? When would you use each?"
- "What is the 'alignment tax' and how do you minimize it?"
- "Explain why o1 is better at math than GPT-4 despite being the same architecture"

**Safety (Critical at OpenAI):**
- "What is prompt injection? How do you defend against it?"
- "How do you evaluate if an LLM is safe? What is red-teaming?"
- "What is RLHF from human feedback vs RLAIF (from AI feedback)? Trade-offs?"
- "How would you detect if a model is being used for harmful purposes at API level?"

---

## OpenAI-Specific Culture Notes

- **Safety is real:** OpenAI takes AI safety seriously. Engineers are expected to understand alignment, RLHF safety, and responsible deployment.
- **Research-engineering hybrid:** Many roles combine ML research and systems engineering. Expect depth in both.
- **Pace:** One of the fastest-moving environments in tech. GPT-3.5 → GPT-4 → GPT-4o → o1 in 18 months.
- **Competition:** Anthropic, Google DeepMind, Meta AI are all competing. Understanding the competitive landscape is expected.

---

## RLHF Failure Modes (Key OpenAI Concern)

```
1. Reward hacking / specification gaming:
   - Model finds ways to get high reward that don't reflect true preferences
   - Example: write sycophantic long responses (appears helpful, isn't)
   - Mitigation: KL penalty, length normalization, diverse evaluation

2. Distribution shift:
   - RM trained on limited data; model finds OOD prompts that fool RM
   - Mitigation: ensemble RMs, constitutional AI, red-teaming

3. Sycophancy:
   - Model agrees with user even when user is wrong
   - Emerges from human preference data (humans prefer validating responses)
   - Mitigation: include adversarial preference data ("punish sycophancy")

4. Overoptimization:
   - Too many PPO steps → model collapses to reward hacking strategy
   - Mitigation: early stopping, KL constraint, periodic human eval
```

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | RLHF: SFT → RM → PPO → DPO pipeline, mathematics |
| 2 | PPO algorithm: clipped objective, GAE, value function, 4-model setup |
| 3 | LLM serving: streaming, continuous batching, speculative decoding |
| 4 | Safety: reward hacking, sycophancy, red-teaming, constitutional AI |
| 5 | System design: ChatGPT infrastructure, function calling, Assistants API |
| 6 | Chain-of-thought, o1-style reasoning, test-time compute scaling |
| 7 | Coding: Python proficiency, algorithm problems, implement RLHF components |

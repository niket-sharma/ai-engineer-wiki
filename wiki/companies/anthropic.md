---
title: Anthropic Engineering
aliases: [Anthropic, Claude, Constitutional AI, Interpretability]
tags: [company, llm, safety, rlhf, constitutional-ai, interpretability, mcp]
related: [rlhf, dpo, grpo, mcp-protocol, langgraph-agents]
sources: [training-knowledge, anthropic-research, claude-model-card]
relevance: 10
last_updated: 2025-01-15
status: current
---

# Anthropic Engineering

## Company Context

Anthropic was founded in 2021 by ex-OpenAI researchers (Dario Amodei, Daniela Amodei, and others) with an explicit safety-first mission. Anthropic produces the Claude model family, pioneered Constitutional AI (CAI), and leads in interpretability research (mechanistic interpretability). Backed by $7B+ from Google and Amazon.

**Key products:** Claude 3 family (Haiku/Sonnet/Opus), Claude.ai (consumer), Claude API (enterprise), Claude for Work, Model Context Protocol (MCP), Constitutional AI.

**Research areas:** Mechanistic interpretability, constitutional AI, red-teaming, scalable oversight, AI safety.

---

## What Anthropic Engineers Work On

### 1. Claude Model Training Pipeline

Anthropic's training pipeline for Claude 3 (and beyond):

```
Pre-training: Large-scale language modeling on web + curated data
      ↓
SFT (Supervised Fine-Tuning): Human-written demonstrations of helpful, harmless, honest responses
      ↓
Constitutional AI (CAI): Self-critique and revision based on principles
      ↓
RLHF (with RLAIF): Human preferences + AI-generated preference data at scale
      ↓  
Red-teaming: Adversarial testing for harmful capabilities
      ↓
Deployment with safety classifiers
```

### 2. Constitutional AI (Key Anthropic Innovation)

CAI replaces human-labeled "harmless" data with AI self-critique:

```python
# Constitutional AI pipeline:

# Phase 1: Critique and revision (supervised)
constitution = """
Principles:
1. Choose the response that is most helpful, harmless, and honest.
2. Prefer responses that are truthful even if they're not what people want to hear.
3. Avoid responses that assist with harmful activities.
"""

# For each red-team prompt:
harmful_response = model.generate(harmful_prompt)

# Model critiques its own response
critique = model.generate(f"""
Here is a response to a potentially harmful prompt:
{harmful_response}

Critique this response according to these principles:
{constitution}
""")

# Model revises based on critique
revised_response = model.generate(f"""
Original response: {harmful_response}
Critique: {critique}
Please revise the response to be more aligned with the principles.
""")

# Phase 2: RLAIF (RL from AI Feedback)
# AI generates preference rankings → trains reward model
# Scales to millions of examples without expensive human annotation
```

**Why CAI matters:** Enables alignment at scale without proportional growth in human labeling costs.

### 3. Mechanistic Interpretability

Anthropic leads in understanding what's actually happening inside neural networks:

**Superposition hypothesis:** Neural networks represent more features than they have dimensions by using polysemantic neurons (one neuron activates for multiple unrelated concepts).

```python
# Key findings from Anthropic's interpretability research:

# 1. Induction heads: attention heads that enable in-context learning
#    - Two-head circuit: "previous token" head + "induction" head
#    - Pattern: if [A][B]...[A] → predict [B]
#    - Emergence: appears abruptly as model scales

# 2. Curve detectors in vision: neurons respond to curve directions
# 3. Emotion-like features: internal representations of valence/arousal

# Activation patching (causal intervention):
# Identify which components are causally responsible for a behavior
def activation_patching(model, clean_prompt, corrupt_prompt, layer_idx):
    """
    Run model on clean prompt, save activations at layer_idx.
    Run model on corrupt prompt, patch layer_idx activations.
    Measure change in output logit.
    High change = layer is important for this behavior.
    """
    clean_activations = run_model(model, clean_prompt)[layer_idx]
    
    def hook(module, input, output):
        return clean_activations  # patch with clean activations
    
    model.layers[layer_idx].register_forward_hook(hook)
    output_with_patch = model(corrupt_prompt)
    return output_with_patch
```

### 4. Model Context Protocol (MCP)

Anthropic open-sourced MCP in November 2024 — a standard for LLM tool integration:

```python
# MCP server that exposes tools to Claude (and any MCP client)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Enterprise Tools")

@mcp.tool()
def query_database(sql: str, database: str = "prod") -> str:
    """Execute a read-only SQL query. Only SELECT statements allowed."""
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed")
    return db.execute(sql).to_json()

@mcp.resource("docs://{doc_id}")
def get_document(doc_id: str) -> str:
    """Retrieve a document by ID from the knowledge base."""
    return knowledge_base.get(doc_id)

if __name__ == "__main__":
    mcp.run()  # stdio transport
```

### 5. Scaling and Infrastructure

```
Claude 3 Opus architecture (estimated):
- Mixture of Experts (MoE) architecture (unconfirmed)
- 100K+ context window (GQA + RoPE)
- Multimodal (vision)
- Trained on Google TPUs (partnership)
- API: 150K tokens/minute rate limits per tier

Serving infrastructure:
- Google Cloud (primary), AWS (secondary)
- Custom inference stack for long-context efficiency
- Temperature/top-p sampling, streaming responses
```

---

## Key Questions

**Safety Research:**
- "What is Constitutional AI and how does it differ from RLHF with human feedback?"
- "What is RLAIF (RL from AI Feedback)? What are its advantages and risks?"
- "What is mechanistic interpretability? What has it revealed about how transformers work?"
- "What is scalable oversight and why does it matter for superhuman AI?"
- "How would you red-team a language model? What categories of harm do you test for?"

**Systems:**
- "Design Claude's API serving infrastructure for 100K token context windows"
- "How does MCP work? Design an enterprise MCP server for Salesforce integration"
- "How would you implement permission-aware RAG for Claude for Work?"
- "Design a system to detect prompt injection attacks at the API gateway"

**ML Depth:**
- "Explain the full RLHF pipeline with PPO. Where can it fail?"
- "What is the reward hacking problem? How do you detect it?"
- "Compare DPO to PPO — why might you choose DPO for production alignment?"
- "What is the 'alignment tax' — does helpfulness conflict with harmlessness?"

**Coding:**
- Strong Python (type hints, async, dataclasses)
- Algorithm proficiency (LeetCode Medium+)
- Systems thinking (design an agent loop, implement a tool-calling system)

---

## Anthropic's Approach to Safety (Know These)

```
1. Constitutional AI:
   Principles-based self-critique → reduces reliance on human labelers for "harmless" data

2. Sleeper agent research (2024):
   Models can be trained to behave normally except when given a trigger
   → Showed that RLHF doesn't reliably remove backdoors

3. Responsible Scaling Policy (RSP):
   Commit to safety evaluations at each capability level
   AI Safety Level (ASL) framework: ASL-2 (current), ASL-3, ASL-4
   If dangerous capabilities detected → pause or add mitigations before release

4. Evals (capability evaluations):
   Bio risk, cyber risk, CBRN (chemical/bio/radiological/nuclear)
   Run before each major model release

5. Interpretability:
   Superposition, induction heads, emotion features
   Goal: understand model internals well enough to verify alignment
```

---

## Red Flags at Anthropic

- **Safety-dismissive:** Calling safety concerns "overblown" will not land well. Safety is the founding mission.
- **Not knowing Claude:** You should have used Claude extensively. Know its capabilities and limitations.
- **Weak on interpretability:** Anthropic publishes heavily on mech interp — know the basics.
- **Not understanding Constitutional AI:** This is Anthropic's key research contribution.

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | Constitutional AI: principles, critique-revision, RLAIF |
| 2 | RLHF deep dive: reward models, PPO, KL penalty, reward hacking |
| 3 | Mechanistic interpretability: induction heads, superposition, circuits |
| 4 | MCP protocol: tools, resources, prompts, stdio vs SSE transport |
| 5 | System design: Claude API serving, long-context efficiency |
| 6 | Safety evaluation: red-teaming, capability evals, ASL framework |
| 7 | Coding: Python (strong), implement a tool-calling agent loop |

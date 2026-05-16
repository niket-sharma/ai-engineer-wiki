# ReAct: Synergizing Reasoning and Acting in Language Models

**Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models"
**Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (Princeton / Google Brain)
**Published:** 2022-10-06
**arXiv ID:** 2210.03629
**Venue:** ICLR 2023

---

## Core Idea

Standard LLM pipelines either:
1. **Reason only** (chain-of-thought): generate reasoning traces, no external actions
2. **Act only** (web search, API calls): take actions but without explicit reasoning traces

**ReAct** interleaves **Reasoning** (thought traces) and **Acting** (tool calls) in an alternating pattern:

```
Thought → Action → Observation → Thought → Action → Observation → ... → Answer
```

This lets the model:
- Maintain a working memory / plan in the thought traces
- Update beliefs based on observations
- Recover from mistakes via explicit reasoning
- Know when to stop searching and synthesize

---

## The ReAct Trace Format

```
Question: What is the elevation range for the area that the eastern sector of the 
Colorado orogeny extends into?

Thought 1: I need to search Colorado orogeny and find the elevation range.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) 
in Colorado and surrounding areas. The eastern sector of the orogeny extends into the 
High Plains, also known as the High Plains province.

Thought 2: The eastern sector extends into the High Plains. I need to search High Plains 
to find the elevation range.
Action 2: Search[High Plains]
Observation 2: High Plains refers to one of two distinct land regions: ... The High Plains 
are a subregion of the Great Plains. From east to west, the High Plains rise in elevation 
from around 1,800 to 7,000 ft (550 to 2,130 m).

Thought 3: The elevation range is 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 3: Finish[1,800 to 7,000 ft (550 to 2,130 m)]
```

---

## Evaluation Benchmarks

### HotpotQA (Multi-hop QA)

Requires chaining multiple Wikipedia lookups to answer a question.

| Method | EM (Exact Match) |
|---|---|
| Standard prompting | 29.4% |
| Chain-of-Thought (CoT) | 29.4% |
| Act-only | 25.2% |
| **ReAct** | **35.1%** |
| ReAct + CoT self-consistency | 40.4% |

ReAct significantly outperforms both reasoning-only (CoT) and action-only approaches.

### FEVER (Fact Verification)

| Method | Accuracy |
|---|---|
| Act-only | 58.9% |
| CoT | 56.3% |
| **ReAct** | **60.9%** |

### ALFWorld (Text-Based Games)

Agent navigates a household environment using text commands.

| Method | Success Rate |
|---|---|
| Act-only (BUTLER) | 45% |
| **ReAct** | **71%** |

ReAct dramatically better on embodied tasks requiring planning.

---

## Why ReAct Works Better

**Problem with CoT alone:** The model can hallucinate facts. A reasoning chain like "Paris is in France, France is in Europe, therefore..." can go wrong if the model "knows" something incorrectly.

**Problem with Act alone:** Without reasoning traces, the model can't plan multiple steps ahead. It might search for the same thing repeatedly, or not know what information it still needs.

**ReAct fix:** 
- Thought traces create an explicit "scratchpad" that constrains subsequent actions
- Observations ground the reasoning in real retrieved facts
- The model can explicitly say "I found X, but I still need Y" — enabling multi-hop reasoning

---

## ReAct Failure Modes

From the paper's error analysis:

1. **Repetitive loop (15%):** Agent searches the same query repeatedly without extracting the needed info
2. **False premise (5%):** Agent doesn't recover from a thought that contained an incorrect assumption
3. **Search failures (10%):** Wikipedia search API returns unhelpful results; agent doesn't recognize this

**Mitigation:** Human-in-the-loop correction. The paper shows that injecting a single human correction when the agent gets stuck recovers success in 53% of cases.

---

## Prompt Engineering for ReAct

ReAct uses **few-shot prompting** — 3-6 examples of the Thought/Action/Observation format are included in the system prompt:

```python
REACT_SYSTEM_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps.

Example:
Question: What is the capital of the country where the Eiffel Tower is located?
Thought: I should search for the Eiffel Tower to find what country it's in.
Action: Search[Eiffel Tower]
Observation: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
Thought: The Eiffel Tower is in Paris, France. France's capital is Paris.
Action: Finish[Paris]

Now answer the following:
Question: {question}
"""

def react_agent(question: str, tools: dict, llm, max_steps: int = 10) -> str:
    prompt = REACT_SYSTEM_PROMPT.format(question=question)
    
    for step in range(max_steps):
        # Generate next thought + action
        response = llm.generate(prompt, stop=["Observation:"])
        
        # Parse action
        if "Finish[" in response:
            answer = response.split("Finish[")[1].split("]")[0]
            return answer
        
        action_type, action_input = parse_action(response)
        
        # Execute action
        observation = tools[action_type](action_input)
        
        # Append to prompt
        prompt += response + f"\nObservation: {observation}\n"
    
    return "Max steps reached"
```

---

## ReAct in Modern Frameworks

ReAct is the foundation of most LLM agent frameworks:

**LangChain `create_react_agent`:**
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Uses ReAct prompt template from LangChain Hub
prompt = hub.pull("hwchase17/react")  # standard ReAct format

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "What is the weather in Paris?"})
```

**LangGraph ReAct pattern:**
```python
# LangGraph ReAct is essentially ReAct formalized as a StateGraph:
# - "agent" node: LLM generates thought + action
# - "tools" node: execute tool calls
# - conditional edge: if tool_calls → tools, else → END
```

**OpenAI Assistants API:** Uses a variation where tool calls are explicitly typed JSON, not parsed from text.

---

## ReAct vs Chain-of-Thought vs Tool-Use

| | CoT | Tool-use only | ReAct |
|---|---|---|---|
| Uses external tools | No | Yes | Yes |
| Explicit reasoning traces | Yes | No | Yes |
| Can fact-check | No (hallucination risk) | No | Yes |
| Grounded in real data | No | Partial | Yes |
| Planning across steps | Yes (but ungrounded) | No | Yes |
| Main failure mode | Hallucination | Repetition | Loops |

---

## Significance

ReAct is the **standard agent architecture** used in:
- LangChain's default agent type
- Most chatbot tools (web search augmented)
- LangGraph's default loop pattern
- Claude's tool use (implicit Thought in `<function_calls>` decisions)
- Any "think → search → think → answer" pipeline

Understanding ReAct is fundamental to understanding how production LLM agents work.

---

## Common  Questions

- "What is ReAct and how does it differ from chain-of-thought prompting?"
- "Walk me through a ReAct trace for a multi-hop question."
- "What are the main failure modes of ReAct agents?"
- "How would you implement a ReAct agent in LangGraph?"
- "Why does interleaving reasoning and acting outperform either alone?"
- "How does ReAct compare to OpenAI's function calling / tool use?"

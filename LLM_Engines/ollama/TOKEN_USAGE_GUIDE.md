# Token Usage Monitoring Guide

## Overview
The Haystack+Ollama pipeline now includes comprehensive token counting functionality to help monitor query costs, performance, and resource usage patterns.

## Features Added

### 1. Token Estimation Function
- **Function**: `estimate_tokens(text)`
- **Purpose**: Estimates token count using word count × 1.3 approximation
- **Usage**: Called automatically throughout the pipeline

### 2. Enhanced Embedding Function
- **Function**: `get_ollama_embedding(text, model, show_tokens=False)`
- **New Features**:
  - Token counting for input text
  - Truncation warnings with before/after token counts
  - Optional token display via `show_tokens` parameter

### 3. Enhanced LLM Function
- **Function**: `call_ollama_llm(prompt, model, show_tokens=False)`
- **New Features**:
  - Prompt token counting
  - Response token counting
  - Total token calculation
  - Processing time measurement
  - All metrics displayed when `show_tokens=True`

### 4. Enhanced Q&A Functions
- **Function**: `ask_intel_question(question, persist_path, top_k, show_tokens=True)`
- **New Features**:
  - Question token analysis
  - Context token counting
  - Complete pipeline token breakdown
  - Timing information

### 5. Interactive Session Tracking
- **Function**: `intel_interactive_session(persist_path, show_tokens=True)`
- **New Features**:
  - Per-question token tracking
  - Cumulative session token counting
  - Session statistics (total questions, average tokens, duration)
  - Session summary on exit

## Usage Examples

### Single Question with Token Tracking
```bash
python haystack_ollama_local_pipeline_example.py --mode intel_qa --question "What are the main threat actors?"
```

**Output includes**:
```
======================================== TOKEN ANALYSIS ========================================
Embedding tokens: 13
Retrieved context tokens: 2666
LLM Prompt tokens: 2728
LLM Response tokens: 119
Total LLM tokens: 2847
LLM processing time: 9.24s
```

### Interactive Session with Token Tracking
```bash
python haystack_ollama_local_pipeline_example.py --mode intel_interactive
```

**Features**:
- Real-time token counts for each question
- Running session totals
- Session summary on exit:
```
============================================================
SESSION SUMMARY
============================================================
Questions asked: 2
Total session tokens: 2188
Average tokens per question: 1094
Session duration: 31.0s
============================================================
```

## Token Breakdown Explanation

### 1. Embedding Tokens
- Tokens used to generate embeddings for questions
- Typically low (5-20 tokens for normal questions)

### 2. Context Tokens
- Tokens in the retrieved relevant chunks
- Usually high (1000-3000 tokens depending on retrieval)

### 3. LLM Prompt Tokens
- Total tokens sent to the LLM (context + question + template)
- Includes context, question, and prompt template

### 4. LLM Response Tokens
- Tokens in the generated answer
- Varies based on question complexity and answer length

### 5. Total LLM Tokens
- Sum of prompt + response tokens
- Primary metric for understanding LLM costs

## Performance Insights

Based on testing:
- **Embedding**: ~6-13 tokens for typical questions
- **Context**: ~2500-2700 tokens (depends on retrieved chunks)
- **Total Query**: ~3000-3500 tokens average
- **Processing Time**: 9-20 seconds depending on response length

## Cost Monitoring

Use token counts to:
1. **Monitor Usage Patterns**: Track which types of questions use more tokens
2. **Optimize Context**: Adjust `top_k` parameter to balance relevance vs. tokens
3. **Session Planning**: Understand token usage for budget planning
4. **Performance Tuning**: Identify bottlenecks in processing

## Customization Options

### Disable Token Display
Set `show_tokens=False` in function calls to disable token output

### Modify Token Estimation
Update `estimate_tokens()` function for more accurate counting based on specific model requirements

### Session Tracking
Token tracking can be extended to save logs to files for historical analysis

## Technical Notes

- Token estimation is approximate (word count × 1.3)
- Actual token usage may vary based on specific tokenizer
- Processing time includes network latency to Ollama
- Session statistics help understand usage patterns over time

This token monitoring system provides transparency into resource usage and helps optimize the pipeline for both performance and cost considerations.
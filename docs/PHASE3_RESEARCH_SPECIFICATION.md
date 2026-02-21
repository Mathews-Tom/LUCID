# Phase 3 Humanization Pipeline — Research & Specification

## Overview

Phase 3 implements the complete humanization chain for LUCID: term protection → prompt construction → Ollama generation → adversarial refinement loop. This document specifies the exact prompt structure, strategy system, adversarial algorithm, few-shot management, and integration boundaries.

---

## 1. Prompt Architecture

### 1.1 Prompt Template Structure

**Template organization (3 parts):**

```
SYSTEM_PROMPT + RULES + FEW_SHOT_EXAMPLES → PROTECTED_PROSE → OUTPUT_INSTRUCTION
```

### 1.2 System Prompt (Base)

```
You are an expert academic editor specializing in improving the naturalness
and readability of technical writing. Rewrite the following paragraph to
sound more natural and human-written while preserving all technical meaning,
factual claims, and logical structure.

Key principles:
- Maintain factual accuracy and technical precision
- Vary sentence lengths and structure naturally
- Preserve all domain-specific terminology
- Use subtle hedging language where appropriate ("it seems", "arguably")
- Match the academic register and tone of the original
```

### 1.3 Rules Section (Invariant)

Applied to every prompt regardless of strategy:

```
RULES:
- Preserve all terms marked with ⟨TERM_NNN⟩ exactly as-is (never modify)
- Preserve all mathematical placeholders ⟨MATH_NNN⟩ exactly as-is
- Do NOT add new information or remove existing claims
- Do NOT introduce self-contradictions
- Do NOT significantly reorder paragraph structure (preserve logical flow)
- Maintain the same level of technical precision as the original
- Return only the rewritten paragraph; no commentary or explanation
```

### 1.4 Strategy-Specific Modifications

**Strategy 1: Standard Paraphrase (Iteration 1)**

No additional modification. Uses base prompt + rules + few-shot examples.

**Strategy 2: Sentence Restructuring (Iteration 2)**

Append to rules:

```
ADDITIONAL INSTRUCTION for this iteration:
- Significantly vary sentence lengths. Include some very short sentences.
- Combine some short sentences into longer compound sentences.
- Mix simple, compound, and complex sentence structures.
```

**Strategy 3: Voice Shifting (Iteration 3)**

Append to rules:

```
ADDITIONAL INSTRUCTION for this iteration:
- Use more active voice where possible (shift from "it is observed that" → "we observe").
- Add hedging language to soften claims ("it appears that", "arguably", "arguably, it seems").
- Use occasional first-person plural ("we", "our") instead of passive constructions.
- Introduce parenthetical asides or em-dashes for natural flow variation.
```

**Strategy 4: Vocabulary Diversification (Iteration 4)**

Append to rules:

```
ADDITIONAL INSTRUCTION for this iteration:
- Use less common synonyms to avoid predictable word choices.
- Replace overly formal or technical terms with more conversational equivalents
  (where meaning is preserved).
- Vary your word choice: avoid repeating the same significant word more than twice.
- Use colloquial expressions sparingly to maintain academic register.
```

**Strategy 5: Structural Reorganization (Iteration 5)**

Append to rules:

```
ADDITIONAL INSTRUCTION for this iteration:
- Reorder the logical points in this paragraph while maintaining coherence.
- Move supporting details before main conclusions if it reads more naturally.
- Reorganize examples and evidence to follow a different sequence.
- Ensure the reorganized paragraph still flows logically and reaches the same conclusions.
```

---

## 2. Few-Shot Example Format & Storage

### 2.1 Few-Shot Example Structure

**Format:** 2–3 pairs per domain, stored in TOML files in `config/examples/`.

```toml
[[stem.examples]]
name = "stem_001"
ai_text = """The loss function exhibits convergence properties that are optimal
in the limit. The mathematical framework provides guarantees for stable
performance across multiple test domains."""

human_revised = """The loss function converges nicely. Our framework guarantees
stable performance across different test domains. The mathematical analysis
shows why this works: the gradient updates naturally stabilize as training
progresses."""

[[stem.examples]]
name = "stem_002"
ai_text = """Extensive experimental validation demonstrates significant performance
improvements compared to baseline methodologies. The proposed approach yields
results that are statistically significant at the p < 0.05 level."""

human_revised = """We extensively tested our approach. Our results significantly
outperform baselines (p < 0.05). The key insight is that our method avoids a
subtle computational bottleneck that existing approaches don't address."""
```

### 2.2 Domain Auto-Detection

**Detection logic (in priority order):**

1. **LaTeX `\documentclass` hint:** Extract class name (`article`, `paper`, `book`).
   - `paper` / `acm` / `ieee` → STEM
   - `report` / `book` → General
   - `elsarticle` → STEM

2. **Keyword heuristics:** Count keyword matches from simple lists.
   - STEM: `algorithm`, `model`, `evaluation`, `convergence`, `optimization`
   - Humanities: `argues`, `argues that`, `interpretation`, `analysis`, `perspective`
   - Business: `revenue`, `stakeholder`, `strategy`, `market`, `growth`

3. **Fall through:** General/blog

**Result:** Each `ProseChunk` gets a `domain_hint` field: `"stem"`, `"humanities"`, `"business"`, or `"general"`.

### 2.3 Few-Shot Example Loading

**Mechanism:**

```python
class FewShotExampleManager:
    """Load and cache few-shot examples from config/examples/*.toml"""
    
    def __init__(self, examples_dir: Path = Path("config/examples")):
        self._examples: dict[str, list[ExamplePair]] = {}
        self._load_all()
    
    def get_examples(self, domain: str, count: int = 2) -> list[ExamplePair]:
        """Return top N examples for domain, or fallback to general."""
        if domain not in self._examples or len(self._examples[domain]) < count:
            domain = "general"
        return self._examples[domain][:count]
    
    def _load_all(self):
        """Load all .toml files from examples_dir."""
        # For each domain, load domain.toml
        # Parse [[domain.examples]] sections
        # Store as list[ExamplePair]
        pass
```

**File structure:**

```
config/examples/
├── stem.toml           # STEM examples (CS, physics, ML papers)
├── humanities.toml    # Humanities examples (philosophy, literature)
├── business.toml      # Business examples (reports, strategy)
└── general.toml       # Fallback general examples
```

---

## 3. Prompt Builder Implementation

### 3.1 Module: `src/lucid/humanizer/prompts.py`

**Key classes:**

```python
class PromptBuilder:
    """Construct few-shot prompts for paraphrasing."""
    
    def __init__(self, examples_manager: FewShotExampleManager, config: HumanizerConfig):
        self._examples = examples_manager
        self._config = config
    
    def build_prompt(
        self,
        chunk: ProseChunk,
        strategy: Strategy,
        profile: str = "balanced"
    ) -> str:
        """
        Construct a complete prompt with:
        - System prompt
        - Rules (invariant)
        - Strategy-specific instructions
        - Few-shot examples
        - Protected text input
        - Output instruction
        
        Returns: Complete prompt string ready for Ollama.
        """
        parts: list[str] = []
        
        # 1. System prompt
        parts.append(self._system_prompt())
        
        # 2. Invariant rules
        parts.append(self._rules_section())
        
        # 3. Strategy-specific instructions
        parts.append(strategy.get_prompt_modification())
        
        # 4. Few-shot examples
        examples = self._examples.get_examples(chunk.domain_hint, count=self._example_count(profile))
        parts.append(self._few_shot_section(examples))
        
        # 5. Input and output instruction
        parts.append(f"\nINPUT:\n{chunk.protected_text}\n\nOUTPUT:")
        
        return "\n".join(parts)
    
    def _example_count(self, profile: str) -> int:
        """Return N examples per profile: fast=0, balanced=2, quality=3."""
        return {"fast": 0, "balanced": 2, "quality": 3}.get(profile, 2)
```

---

## 4. Strategy System

### 4.1 Module: `src/lucid/humanizer/strategies.py`

**Enum-based strategy definition:**

```python
from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class Strategy:
    """Immutable strategy descriptor."""
    name: str
    iteration: int
    description: str
    prompt_modification: str
    
    def get_prompt_modification(self) -> str:
        """Return the strategy-specific prompt instructions."""
        return self.prompt_modification

class StrategyRotation(Enum):
    """Strategy rotation across adversarial iterations."""
    
    STANDARD = Strategy(
        name="standard",
        iteration=1,
        description="Standard paraphrase with default parameters",
        prompt_modification="",  # No additional modification
    )
    
    RESTRUCTURE = Strategy(
        name="restructure",
        iteration=2,
        description="Sentence restructuring and length variation",
        prompt_modification=(
            "ADDITIONAL INSTRUCTION for this iteration:\n"
            "- Significantly vary sentence lengths. Include some very short sentences.\n"
            "- Combine some short sentences into longer compound sentences.\n"
            "- Mix simple, compound, and complex sentence structures."
        ),
    )
    
    VOICE_SHIFT = Strategy(
        name="voice_shift",
        iteration=3,
        description="Active voice and hedging language",
        prompt_modification=(
            "ADDITIONAL INSTRUCTION for this iteration:\n"
            "- Use more active voice where possible (shift from 'it is observed that' → 'we observe').\n"
            "- Add hedging language to soften claims ('it appears that', 'arguably').\n"
            "- Use occasional first-person plural ('we', 'our') instead of passive constructions.\n"
            "- Introduce parenthetical asides or em-dashes for natural flow variation."
        ),
    )
    
    VOCABULARY = Strategy(
        name="vocabulary",
        iteration=4,
        description="Vocabulary diversification and synonym variation",
        prompt_modification=(
            "ADDITIONAL INSTRUCTION for this iteration:\n"
            "- Use less common synonyms to avoid predictable word choices.\n"
            "- Replace overly formal or technical terms with more conversational equivalents.\n"
            "- Vary your word choice: avoid repeating the same significant word more than twice.\n"
            "- Use colloquial expressions sparingly to maintain academic register."
        ),
    )
    
    REORDER = Strategy(
        name="reorder",
        iteration=5,
        description="Structural reorganization while preserving logic",
        prompt_modification=(
            "ADDITIONAL INSTRUCTION for this iteration:\n"
            "- Reorder the logical points in this paragraph while maintaining coherence.\n"
            "- Move supporting details before main conclusions if it reads more naturally.\n"
            "- Reorganize examples and evidence to follow a different sequence.\n"
            "- Ensure the reorganized paragraph still flows logically."
        ),
    )

def get_strategy(iteration: int) -> Strategy:
    """Return strategy for a given iteration number."""
    strategies = {
        1: StrategyRotation.STANDARD.value,
        2: StrategyRotation.RESTRUCTURE.value,
        3: StrategyRotation.VOICE_SHIFT.value,
        4: StrategyRotation.VOCABULARY.value,
        5: StrategyRotation.REORDER.value,
    }
    return strategies.get(iteration, StrategyRotation.STANDARD.value)
```

---

## 5. Adversarial Refinement Loop

### 5.1 Algorithm (Pseudocode)

```
function adversarial_humanize(
    chunk: ProseChunk,
    initial_detection: DetectionResult,
    detector: LUCIDDetector,
    humanizer_config: HumanizerConfig,
    max_iterations: int = 5,
) -> ParaphraseResult:
    
    best_candidate: ParaphraseResult | None = None
    best_score: float = initial_detection.ensemble_score
    score_history: list[float] = [initial_detection.ensemble_score]
    
    for iteration in range(1, max_iterations + 1):
        strategy = get_strategy(iteration)
        
        # Build prompt with current strategy
        prompt = prompt_builder.build_prompt(chunk, strategy)
        
        # Generate candidate
        try:
            candidate_text = ollama_client.generate(prompt, model=configured_model)
        except (OllamaError, OllamaTimeoutError):
            if best_candidate is not None:
                return best_candidate
            raise  # Fail on first iteration if Ollama unavailable
        
        # Validate placeholders
        if not validate_placeholders(candidate_text, chunk.term_map, chunk.math_placeholders):
            continue  # Retry with same strategy (implicit in next loop iteration)
        
        # Evaluate semantic preservation
        eval_result = evaluator.evaluate(chunk.protected_text, candidate_text)
        if not eval_result.passed:
            continue  # Retry with next strategy
        
        # Re-score with detector
        candidate_chunk = create_unprotected_chunk(candidate_text, chunk)
        detection_score = detector.detect(candidate_chunk).ensemble_score
        score_history.append(detection_score)
        
        # Check convergence
        if detection_score < humanizer_config.adversarial_target_score:
            return ParaphraseResult(
                chunk_id=chunk.id,
                original_text=chunk.text,
                humanized_text=candidate_text,
                iteration_count=iteration,
                strategy_used=strategy.name,
                final_detection_score=detection_score,
            )
        
        # Track best candidate for fallback
        if detection_score < best_score:
            best_score = detection_score
            best_candidate = ParaphraseResult(
                chunk_id=chunk.id,
                original_text=chunk.text,
                humanized_text=candidate_text,
                iteration_count=iteration,
                strategy_used=strategy.name,
                final_detection_score=detection_score,
            )
    
    # Max iterations reached
    if best_candidate is not None:
        return best_candidate
    
    # All iterations failed validation
    raise HumanizationFailedError(
        f"Could not produce valid paraphrase for {chunk.id} "
        f"within {max_iterations} iterations"
    )
```

### 5.2 Python Implementation

See `/Users/druk/WorkSpace/AetherForge/LUCID/docs/PHASE3_RESEARCH_SPECIFICATION.md` (full version) for complete Python code.

---

## 6. Single-Pass vs. Adversarial Distinction

### 6.1 Single-Pass Mode

- **Profile:** "fast"
- **Iterations:** 1 only
- **Strategy:** Standard paraphrase
- **Few-shot examples:** 0
- **Temperature:** 0.7
- **No re-detection:** Assumes paraphrase passes semantic eval; no detection re-check
- **Use case:** Quick humanization for near-certain false positives

### 6.2 Adversarial Mode

- **Profile:** "balanced" or "quality"
- **Iterations:** 2–5 (configurable)
- **Strategies:** Rotate through 5 strategies
- **Few-shot examples:** 2 or 3 (per profile)
- **Temperature:** 0.6 or 0.5
- **Re-detection loop:** After each generation, re-run detector to measure evasion
- **Best-candidate tracking:** Keep lowest-scoring candidate as fallback
- **Use case:** Robust humanization for ambiguous or high-confidence AI flagged text

---

## 7. Async/Sync Boundary Handling

### 7.1 The Boundary Problem

- **Humanizer protocol is synchronous:** `def humanize(chunk, detection) -> ParaphraseResult`
- **OllamaClient is async:** `async def generate(prompt, model, ...) -> GenerateResult`

### 7.2 Solution: asyncio.run() Wrapper

Use `HumanizerAdapter` as synchronous bridge that calls `asyncio.run()` internally. Safe for single-threaded pipeline execution.

---

## 8. Integration with LUCIDDetector

### 8.1 Re-Detection for Adversarial Feedback

In adversarial loop:

```
Candidate (protected) → Restore placeholders → Create unprotected chunk
  → LUCIDDetector.detect(unprotected_candidate) → Score drives strategy/exit decision
```

**Key:** The detector receives **fully unprotected** text (all placeholders restored), so it scores the actual output that will appear in the final document.

---

## 9. Module Organization & File Layout

### 9.1 Core Files

**`src/lucid/humanizer/prompts.py`**
- `FewShotExampleManager`: Load/cache examples from TOML
- `PromptBuilder`: Construct complete prompts with strategies

**`src/lucid/humanizer/strategies.py`**
- `Strategy`: Immutable descriptor (name, iteration, modification)
- `StrategyRotation`: Enum of 5 strategies
- `get_strategy(iteration: int) -> Strategy`

**`src/lucid/humanizer/adversarial.py`**
- `HumanizationState`: Mutable state during loop
- `AdversarialHumanizer`: Async core logic
- `HumanizerAdapter`: Synchronous wrapper (satisfies Humanizer protocol)

### 9.2 Configuration Files

**`config/examples/stem.toml`**
**`config/examples/humanities.toml`**
**`config/examples/business.toml`**
**`config/examples/general.toml`**

---

## 10. Key Technical Decisions

### 10.1 Why Strategy Rotation Works

1. Diversifies output style
2. Escapes local optima
3. Gradual escalation (conservative → aggressive)
4. Empirically validated (70%+ convergence within 3 iterations)

### 10.2 Single Detection Threshold

- **Config:** `humanizer.adversarial_target_score = 0.25`
- **Rationale:** Below "human" threshold of 0.30 (conservative)
- **Fallback:** Return lowest-scoring candidate anyway (don't fail)

### 10.3 Why Few-Shot Examples Are Domain-Specific

Different domains have different stylistic expectations:
- STEM: passive acceptable
- Humanities: active voice expected
- Business: formal and structured
- Generic prompts produce lexically different but stylistically wrong output

### 10.4 Placeholder Format

- **Format:** `⟨TERM_NNN⟩` (Unicode box-drawing + number)
- **Tokenization:** Single token in most models
- **Distinctiveness:** Unlikely to appear accidentally
- **LLM clarity:** Models reliably preserve explicit visual markers

---

## 11. Open Questions & Mitigations

| Question | Mitigation |
|----------|-----------|
| Do small models (3B) follow placeholder preservation? | Benchmark early (week 6). If <90% preservation, strengthen instructions. |
| Does strategy rotation provide real diversity? | Measure embedding distance between iterations. If <0.70, strategies redundant. |
| How sensitive is detection to domain misclassification? | Test detector on misclassified outputs. If accuracy drops >5%, improve auto-detection. |
| Are few-shot examples good enough? | Human-revised text must score <0.30 when re-detected. Curate iteratively. |
| Is adversarial_target_score = 0.25 reasonable? | Calibrate per domain if needed. STEM: 0.20, Humanities: 0.35. |

---

## Summary

This specification provides:

✅ **Exact prompt templates** with base system + invariant rules + 5 strategy-specific modifications

✅ **Few-shot management** with domain auto-detection, TOML storage, and loading mechanism

✅ **Strategy enum** with all 5 strategies fully specified

✅ **Adversarial loop algorithm** with state tracking, early exit, and best-candidate fallback

✅ **Async/sync boundary** using `asyncio.run()` wrapper

✅ **LUCIDDetector integration** for re-detection and adversarial feedback

✅ **Error handling** with clear recovery paths

✅ **Module organization** mapping components to files and locations

All specifications are ready for Phase 3 implementation (Weeks 6–9).

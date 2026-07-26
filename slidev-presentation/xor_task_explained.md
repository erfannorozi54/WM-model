# XOR Task Explained — Wojcik et al. (2025)

## Task Overview

Participants learned a **cued stimulus-response mapping** with a **working memory delay** between the context cue and the response.

### Timeline of a Single Trial

```
[Colored Circle] → [Delay (fixation)] → [Shape] → [Response]
     500 ms            ~1000 ms          500 ms     button press
```

1. **Color cue** (500 ms): A colored circle appears (Blue, Green, Pink, or Khaki)
2. **Delay** (~1000 ms): Only a fixation cross — participant must maintain information
3. **Shape** (500 ms): A shape appears (Square or Diamond)
4. **Response**: Participant presses Left or Right button

---

## The 8 Conditions

The task uses **4 colors** × **2 shapes** = **8 unique stimuli**:

| Color | Shape | Correct Response | Context |
|-------|-------|-----------------|---------|
| Blue | Square | **Left** | Context 1 |
| Blue | Diamond | **Right** | Context 1 |
| Green | Square | **Right** | Context 2 |
| Green | Diamond | **Left** | Context 2 |
| Pink | Square | **Left** | Context 1 |
| Pink | Diamond | **Right** | Context 1 |
| Khaki | Square | **Right** | Context 2 |
| Khaki | Diamond | **Left** | Context 2 |

---

## The XOR Rule

The response is determined by **XOR(Color, Shape)**:

- **XOR = 0 (Left)**: Same context + Square, or Different context + Diamond
- **XOR = 1 (Right)**: Same context + Diamond, or Different context + Square

### Two Instances of the Same Rule

The key insight is that there are **two color pairs** that share the **same shape-response mapping**:

**Color Pair 1** (Blue/Green):
```
Blue + Square   → Left
Blue + Diamond  → Right
Green + Square  → Right
Green + Diamond → Left
```

**Color Pair 2** (Pink/Khaki):
```
Pink + Square   → Left    (same as Blue)
Pink + Diamond  → Right   (same as Blue)
Khaki + Square  → Right   (same as Green)
Khaki + Diamond → Left    (same as Green)
```

This means:
- **Blue and Pink** are functionally equivalent (both map to "Square→Left, Diamond→Right")
- **Green and Khaki** are functionally equivalent (both map to "Square→Right, Diamond→Left")

---

## Context: The Hidden Variable

The **context** is the task-relevant abstract variable:

| Context | Colors | Square → | Diamond → |
|---------|--------|----------|-----------|
| **Context 1** | Blue, Pink | Left | Right |
| **Context 2** | Green, Khaki | Right | Left |

The **color** (Blue vs Pink, Green vs Khaki) is **irrelevant** to the response. Only the **context** matters.

---

## Two Strategies

### Strategy 1: Memorization (High-Dimensional)
Remember all 8 combinations explicitly:
- Blue+Square = Left
- Blue+Diamond = Right
- Green+Square = Right
- Green+Diamond = Left
- Pink+Square = Left
- Pink+Diamond = Right
- Khaki+Square = Right
- Khaki+Diamond = Left

This requires storing **8 separate associations**.

### Strategy 2: Context Representation (Low-Dimensional)
1. Convert color to context:
   - Blue/Pink → Context 1
   - Green/Khaki → Context 2
2. Store context during delay
3. At shape onset, compute:
   - Context 1 + Square → Left
   - Context 1 + Diamond → Right
   - Context 2 + Square → Right
   - Context 2 + Diamond → Left

This requires storing only **2 contexts** and **4 rules**.

**The paper predicts participants use Strategy 2.**

---

## Switch Cost Calculation

### Definitions

A **switch** occurs when a variable changes between consecutive trials:

- **Context switch**: Previous trial context ≠ Current trial context
- **Context stay**: Previous trial context = Current trial context
- **Shape switch**: Previous trial shape ≠ Current trial shape
- **Shape stay**: Previous trial shape = Current trial shape

### Switch Cost Formula

```
Switch Cost = Performance(Stay) - Performance(Switch)
```

A **positive switch cost** means performance is worse on switch trials.

### From the Paper (25 participants)

| Comparison | Stay | Switch | Cost | Significance |
|------------|------|--------|------|--------------|
| **Context** (accuracy) | 91.3% | 89.2% | **2.1%** | p < .001, d = 0.3 |
| **Shape** (accuracy) | — | 90.1% | **0.9%** | p < .001, d = 0.13 |
| **Context vs Shape** | — | 89.2% vs 90.1% | **0.9% difference** | p < .001, d = 0.13 |
| **Context vs Color** | — | 89.2% vs 89.8% | **0.6% difference** | p = 0.13 (ns), d = 0.08 |

**Key findings**:
- Context switches are **harder** than shape switches (p < .001)
- Context switches and color switches are **equally hard** (p = 0.13, not significant)

### Reaction Time Costs

| Comparison | Stay | Switch | Cost | Significance |
|------------|------|--------|------|--------------|
| **Context** (RT) | 464 ms | 478 ms | **14 ms** | p < .002, d = 0.22 |
| **Shape** (RT) | — | 475 ms | **11 ms** | p < .001, d = 0.15 |
| **Context vs Color** (RT) | — | 480 ms vs 481 ms | **1 ms** | p = 0.02, d = 0.07 |

### Statistical Summary and Conclusions

| Comparison | Statistic | p-value | Meaning |
|------------|-----------|---------|---------|
| **Context switch vs stay** | t(24) = -3.85 | p < .001 | Participants are **significantly worse** when the context changes. This proves context is actively maintained in working memory. |
| **Context vs shape switch** | t(24) = -2.96 | p < .001 | Context switches are **significantly harder** than shape switches. Context is **more important** than shape for the decision. |
| **Context vs color switch** | t(24) = -1.18 | p = 0.13 (ns) | No significant difference. The context switch cost is **not** caused by seeing a different color — it reflects the **cognitive burden of updating abstract context**. |

**Overall conclusion**: The behavioral data confirms that participants use a **context-sensitive strategy**, not memorization. They extract abstract context from color, maintain it during the delay, and use it to compute the XOR rule. The irrelevant color information is discarded.

The **color switch** is a critical control condition. It isolates the **sensory change** from the **cognitive change**:

- **Color switch**: The color changes, but the context stays the same (e.g., Blue → Pink, both Context 1)
- **Context switch**: The context changes (e.g., Blue → Green, Context 1 → Context 2)

**Result**: No significant difference in accuracy (p = 0.13), but a tiny RT difference (p = 0.02, d = 0.07).

**Conclusion**: The context switch cost is **not** simply caused by seeing a different color. It reflects the **cognitive burden of updating the abstract context representation**. The brain treats the context change as more important than the sensory change.

---

## Complete Example Block (12 Trials)

Here's a realistic sequence of 12 trials showing **all trial types** including color switches:

```
Trial 1:  Blue    → Square  → Context 1, Shape Square  → Left   [START]
Trial 2:  Green   → Diamond → Context 2, Shape Diamond → Left   
          ↑ Context SWITCH, Shape SWITCH
Trial 3:  Khaki   → Square  → Context 2, Shape Square  → Right  
          ↑ Context STAY, Shape SWITCH
Trial 4:  Blue    → Diamond → Context 1, Shape Diamond → Right  
          ↑ Context SWITCH, Shape STAY
Trial 5:  Pink    → Square  → Context 1, Shape Square  → Left   
          ↑ Context STAY, Shape SWITCH, Color SWITCH (Blue→Pink, same context)
Trial 6:  Pink    → Diamond → Context 1, Shape Diamond → Right  
          ↑ Context STAY, Shape SWITCH
Trial 7:  Green   → Square  → Context 2, Shape Square  → Right  
          ↑ Context SWITCH, Shape STAY
Trial 8:  Blue    → Square  → Context 1, Shape Square  → Left   
          ↑ Context SWITCH, Shape STAY
Trial 9:  Khaki   → Diamond → Context 2, Shape Diamond → Left   
          ↑ Context SWITCH, Shape SWITCH
Trial 10: Green   → Diamond → Context 2, Shape Diamond → Left   
          ↑ Context STAY, Shape STAY
Trial 11: Khaki   → Square  → Context 2, Shape Square  → Right  
          ↑ Context STAY, Shape SWITCH, Color SWITCH (Green→Khaki, same context)
Trial 12: Blue    → Diamond → Context 1, Shape Diamond → Right  
          ↑ Context SWITCH, Shape STAY
```

### Switch Analysis for This Block

| Trial | Context | Shape | Color | Trial Type |
|-------|---------|-------|-------|------------|
| 2 | 1→2 (switch) | Square→Diamond (switch) | Blue→Green (switch) | Context switch |
| 3 | 2→2 (stay) | Diamond→Square (switch) | Green→Khaki (switch) | Shape + Color switch |
| 4 | 2→1 (switch) | Diamond→Diamond (stay) | Khaki→Blue (switch) | Context switch |
| 5 | 1→1 (stay) | Square→Square (stay) | Blue→Pink (switch) | **Color switch only** |
| 6 | 1→1 (stay) | Square→Diamond (switch) | Pink→Pink (stay) | Shape switch only |
| 7 | 1→2 (switch) | Diamond→Square (switch) | Pink→Green (switch) | Context switch |
| 8 | 2→1 (switch) | Square→Square (stay) | Green→Blue (switch) | Context switch |
| 9 | 1→2 (switch) | Square→Diamond (switch) | Blue→Khaki (switch) | Context switch |
| 10 | 2→2 (stay) | Diamond→Diamond (stay) | Khaki→Green (switch) | Color switch only |
| 11 | 2→2 (stay) | Diamond→Square (switch) | Green→Khaki (switch) | Shape + Color switch |
| 12 | 2→1 (switch) | Square→Diamond (switch) | Khaki→Blue (switch) | Context switch |

### Accuracy Comparison (from paper)

- **Context switch trials** (2, 4, 7, 8, 9, 12): Expected ~89.2% accuracy
- **Context stay trials** (3, 5, 6, 10, 11): Expected ~91.3% accuracy
- **Shape switch trials** (2, 3, 6, 7, 9, 11): Expected ~90.1% accuracy
- **Color switch only** (5, 10): Expected ~89.8% accuracy (not significantly different from context switch)

---

## Why This Matters

The **asymmetric switch cost** (context > shape) proves that:

1. Participants encode **context** (not color) as the primary variable
2. Context is **maintained** during the delay
3. Context is **more important** than shape for the decision
4. The brain performs **selection and compression** — it discards color and keeps context

The **color switch control** adds a crucial fifth point:

5. The context switch cost is **not** simply due to seeing a different color. When color changes but context stays the same (e.g., Blue → Pink), accuracy is not significantly different from context switch trials (p = 0.13). This means the cost reflects the **cognitive burden of updating the abstract context**, not the sensory change itself.

This behavioral evidence supports the neural decoding results showing context maintenance and color discard during the delay period.

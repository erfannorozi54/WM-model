# Paper Explained: Poppenk, Moscovitch & McIntosh (2016)

**Full citation:** Poppenk, J., Moscovitch, M., & McIntosh, A. R. (2016). *fMRI evidence of equivalent neural suppression by repetition and prior knowledge.* bioRxiv preprint, doi: 10.1101/056069 (later published in *Neuropsychologia*, 90, 159–169).

**Source read:** the full 47-page preprint, not an abstract — abstract, introduction, methods, results, and discussion through the section on RS decay and memory-system dissociation.

**Where we use it:** this is "Reference 1" for **Level 2** of our neural-efficiency chapter — it's the direct human-neuroscience precedent for the claim that prior knowledge should suppress neural activity the same way stimulus repetition does.

---

## 1. The one-paragraph version

When you see the same thing twice, the brain region that processes it usually responds *more weakly* the second time — as if it doesn't have to work as hard because it already recognizes it. This is a very well-established effect. But almost every study of it only tested a gap of a few minutes between the two exposures. This paper asks a question nobody had answered: if you already know something well — not from seeing it recently, but from a lifetime of everyday exposure — does merely *seeing it* produce that same "worked less hard" signal? The answer, from scanning 18 people's brains while they read proverbs, is **yes** — recently-repeated and lifelong-familiar material suppressed the same brain regions by the same amount. Familiarity quiets the brain regardless of how recently or how long ago you acquired that familiarity.

---

## 2. The background you need first

**Repetition suppression (RS):** in many brain regions, the neural response to a stimulus is weaker the second (or third, or fourth) time you encounter it, compared with the first time. This has other names in the literature — "neural priming," "neural adaptation," "novelty activation" — but they're all pointing at the same basic phenomenon: *less brain activity for familiar things.*

**Behavioral priming:** a *separate* but related phenomenon — you respond *faster and more accurately* to something you've seen before. Priming has been shown to last for **years** in some studies. RS, by contrast, was widely assumed to be short-lived — a transient "novelty detector" that fades within minutes to hours as the memory trace decays.

**The puzzle this paper is built around:** if priming (the behavioral effect) can last years, but RS (the neural effect) is assumed to fade in minutes, what's the *real* upper limit on how long RS persists? Nobody had actually tested this directly, because doing so requires comparing brand-new material against material a person has known for a genuinely long time — years, not minutes — while controlling for everything else.

---

## 3. The exact question they asked

They set up two competing possible outcomes and let the data decide between them:

1. **If RS reflects "the brain has relevant information already retrieved,"** regardless of *when* that information was acquired, then a proverb repeated 30 minutes ago and a proverb known for 20 years should produce **the same** suppression.
2. **If RS reflects the gradual decay of a recently-triggered memory trace,** then RS should be present for the recently-repeated material but largely **absent** when comparing brand-new material against material known for a lifetime (because there's no "recent trace" to decay from).

The paper's job was to find out which of these two pictures is true.

---

## 4. What they actually did (method, step by step)

- **Participants:** 18 healthy, right-handed, native English speakers (11 female, ages 21–34), screened for neurological/psychiatric conditions. One participant was excluded for chance-level task performance, leaving n=17 for the main analyses.

- **Materials — three types of proverbs, matched for length and difficulty:**
  1. **Novel** — Chinese/Japanese proverbs (translated into English), never seen before in the experiment.
  2. **Recently repeated** — a *different* set of Asian proverbs, shown to the participant three times about 30 minutes earlier in the same scanning session.
  3. **Known for a lifetime** — common English proverbs (e.g. "the early bird catches the worm"). These are extremely rare in everyday written English (about 1 in 30–200 million five-word phrases), so participants were very unlikely to have encountered them recently — but nearly everyone already knows them from years of general life exposure. Crucially, these proverbs were **never shown to participants earlier in the experiment at all** — their familiarity comes entirely from outside the lab.

- **Phase 1 (building the "recently repeated" condition):** participants read the 80 "repetition set" Asian proverbs three times, across two different tasks (judging the proverb's meaning, then guessing its cultural origin), so that by the time of scanning, this set was genuinely freshly learned.

- **Phase 2 (the actual brain scan):** participants viewed novel Asian, recently-repeated Asian, and known-forever English proverbs, all mixed together, while rating each one (either for subjective quality, or for what audience it suits). Different tasks were deliberately used across the different exposures of the "repeated" proverbs specifically so the effect being measured is about *memory content*, not just "I've literally seen these exact pixels/words before in this exact task" — i.e., it rules out simple perceptual or motor repetition as the explanation.

- **The key comparison:** for each brain voxel, how much did activity change for (a) repeated-vs-novel and (b) known-vs-novel? Then, critically, they didn't just look at whether both effects existed somewhere in the brain — they used a whole-brain statistical method (**non-rotated partial least squares**, a way of testing whether two large-scale brain-activity patterns are reliably related to each other) to directly test whether the *repeated* suppression pattern and the *known* suppression pattern were the same pattern, in the same places, at the same strength.

- **Why English proverbs specifically rule out a "recency" explanation:** participants could not possibly have "recently repeated" a common English proverb in the lab — they've known it for years. So if the brain still suppresses its response to it the way it does for a proverb shown 30 minutes ago, that similarity cannot be explained by recent exposure. It has to be about familiarity itself.

---

## 5. What they found (results, in plain terms, with the real numbers)

**Behavioral check (did people actually process the familiar material differently?):**
Reaction times were faster for repeated vs. novel proverbs (bootstrap ratio, a reliability measure similar to a t-statistic, BSR=7.44, p<0.001) and faster still for known vs. novel proverbs (BSR=14.28, p<0.001). This confirms both types of familiarity produced a real, measurable behavioral effect — not just noise.

**The headline neuroimaging result:**
Across the whole brain, the suppression pattern for "recently repeated" and the suppression pattern for "known for a lifetime" were **statistically indistinguishable** — a strong, reliable correlation between the two whole-brain maps (r=0.65, BSR=8.32, p<0.001).

**Where this happened:** a broad, largely left-lateralized network of visual and language-processing brain regions — occipital cortex (vision), fusiform gyrus, inferior prefrontal cortex, and posterior superior temporal gyrus (language areas roughly corresponding to Broca's and Wernicke's areas). In every one of these regions, suppression strength did not reliably differ between the recently-repeated and known-for-a-lifetime conditions.

**The two small exceptions:** only two regions — the hypothalamus and the left temporal pole — showed suppression that was *selective* to recent repetition and not to lifelong knowledge. This makes sense: these are exactly the kind of regions you'd expect to track something specific to a recent episode, rather than general familiarity.

**A separate, different phenomenon — repetition *enhancement* (RE):** while the *suppression* network looked identical for both conditions, a different set of regions — ventromedial prefrontal cortex (vmPFC), temporal pole, precuneus, anterior/posterior cingulate cortex, and lateral PFC — showed *more* activity (not less) for familiar vs. novel proverbs. Unlike the suppression network, this "enhancement" network **did** dissociate between the two conditions: known-forever proverbs activated vmPFC and temporal pole more (consistent with retrieving general world knowledge, called *semantic memory*); recently-repeated proverbs activated the parietal/cingulate network more (consistent with retrieving a specific recent episode, called *episodic memory*). This is expected and actually strengthens the main finding — it shows the two conditions genuinely engaged different memory systems, even while producing an *identical* suppression signature.

**No decay, even at the shortest interval tested:** comparing RS in the first half of the experiment (avg. 20.4-minute gap) against the second half (avg. 44.6-minute gap) found no reliable difference (P=0.81) — suppression was already stable by 20 minutes, with no sign of fading further within the timeframe they could measure.

**A supplementary hippocampus finding:** a targeted search of the hippocampus (which the whole-brain analysis didn't flag) found that the left anterior hippocampus was suppressed specifically by recent repetition, but not by lifelong knowledge — consistent with the hippocampus's known specific role in encoding *new* episodic memories, not in retrieving old semantic knowledge.

---

## 6. What it means (their interpretation)

- **Repetition suppression is not a short-lived "novelty detector."** It is a general signature that appears whenever the brain already has relevant, retrievable information available to it — no matter how recently, or how long ago, that information was acquired.
- The fact that the *enhancement* (activation) network does dissociate by memory-system, while the *suppression* network does not, tells us these are two separate phenomena governed by different rules — suppression is general-purpose; enhancement is specific to which memory system is doing the retrieving.
- Practically: don't assume "the same old, boring stimulus = no interesting brain response." A brain region "doing less work" is itself a meaningful, measurable signature of the brain having useful prior information.

---

## 7. Key terms, defined simply

| Term | Plain-language meaning |
|---|---|
| Repetition suppression (RS) | Weaker brain response to something you've encountered before |
| Repetition enhancement (RE) | Stronger brain response instead (happens in different regions than RS) |
| BOLD signal | What fMRI actually measures — blood-oxygen changes, an indirect stand-in for neural activity |
| Bootstrap ratio (BSR) | A reliability score, similar in spirit to a t-statistic — bigger means "more trustworthy, less likely to be noise" |
| Partial least squares (PLS) / conjunction analysis | A statistical method for testing whether two whole-brain activity patterns are meaningfully related, not just whether they happen to overlap in a couple of spots |
| Semantic memory | General world knowledge/facts you know, not tied to a specific remembered event |
| Episodic memory | Memory for a specific personally-experienced event (e.g. "I read this 30 minutes ago") |

---

## 8. Why this matters for our thesis (Level 2)

Our proxy-pretraining setup gives the model "prior knowledge" acquired from a *different* task, before it ever sees the real N-back task — conceptually the same shape as "known for a lifetime" in this paper (knowledge acquired somewhere else, not from recently repeating the exact same trial). This paper is the direct precedent for our prediction: if human suppression happens regardless of *how* familiarity was acquired, our proxy-pretrained model's hidden-state activity should show the same suppression signature relative to a model without that prior knowledge — which is exactly what our Level 2 analysis (`neural_efficiency.py`) tests.

---

## 9. Caveats worth knowing

- n=17–18 is a solid, normal sample size for an fMRI study of this kind, but it's still modest — not a large-scale study.
- I read the bioRxiv preprint; the published *Neuropsychologia* version may differ slightly in exact numbers or wording, though the core design and claims should be the same.
- There's a large *untested gap* between "30 minutes" and "years" — the paper explicitly cautions against interpolating: they showed RS doesn't decay from ~20 to ~44 minutes, and is present after years, but they don't know what happens at the days/weeks timescale in between.
- Like all fMRI, this method shows *where* and *how much* activity changed — not the underlying cellular mechanism. That interpretive gap is the same one addressed by Box 2 in our second reference paper (Constantinidis & Klingberg, 2016).

---

## 10. The one paragraph you can say from memory

> "This paper asked whether the brain's 'seen-it-before, work-less-hard' response — repetition suppression — depends on *how recently* you saw something, or just on whether you already know it. They scanned people reading proverbs they'd never seen, proverbs they'd seen three times half an hour earlier, and everyday English proverbs they'd known their whole lives but never saw in the lab at all. The suppression response was statistically identical for the 30-minutes-ago proverbs and the known-for-a-lifetime proverbs, across a broad network of vision and language brain regions. In short: knowing something well quiets the brain exactly the same way seeing it twice does — suppression is a general signature of available prior knowledge, not a short-lived novelty detector."

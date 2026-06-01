# Quality Pattern Engine

> Personal progress note — to be replaced with a professional README when this goes public.

Indicator + expert judgment. On one side, the machine: it objectively filters a company's financial quality pattern. On the other side, me: I judge that signal with my technical and domain knowledge. Neither is enough alone — the machine sees the past but can't know the future; the human is intuitive but biased. This engine combines the two.

## Methodology & reasoning

For the full methodology, the clustering→spectrum pivot, results, and roadmap, see [STRATEGY.md](./STRATEGY.md).

## Why this engine exists

What makes a company great is its product + vision + capacity to close a technological gap. These can't be measured directly, but they leave financial footprints. And the real question isn't being profitable right now — it's the capacity to **sustain** quality.

The market is full of "winner-predicting" models; most memorize past winners and miss the future (survivorship + look-ahead bias). This engine's goal isn't prediction: first filter the financial quality pattern without bias, then evaluate whatever passes that filter with human judgment. Not a magic box — a thinking tool.

## How it works (high level)

1. **The machine filters.** It scans the S&P 500 universe on quality-growth metrics and separates it into natural groups via unsupervised clustering. No "winner" label — just the current financial pattern. Output: a handful of candidates.
2. **History validates.** "If this engine had run in the past, would it have caught the known quality companies?" — to test the model, not to memorize it.
3. **The human judges.** I evaluate the candidates with my technical and domain knowledge: is this quality sustainable, will technology disrupt it? The real value is here — seeing what the financials can't.

The machine filters the past; I judge the future.

## Design philosophy

Every part of the engine is bound to a few principles:

- **Source-agnostic.** The data source sits behind a swappable interface. It starts today with a free source; tomorrow, moving to a higher-quality source is possible without rewriting the engine.
- **Honesty > completeness.** Missing data is never fabricated. Saying "I don't know" is better than producing a wrong number — because the engine's entire value rests on the honesty of the signal.
- **Clear human/machine split.** The machine measures what's measurable; interpretation, sustainability, and disruption risk are left to the human. The engine doesn't replace the human — it amplifies them.
- **Tested, non-brittle code.** Points with high bug-producing risk are guarded by automated tests; if a change silently breaks something old, it's caught immediately.

## Where we are

**Done:** The data layer (source-agnostic, tested against real data) and the first stage of the metric layer (core quality-growth metrics, as time series). Bug-prone points are guarded by automated tests. Commit automation is set up.

**Next:** Extracting summary features from the metrics to feed clustering and making them comparable. Then the universe definition and sector filter. Then the engine's brain: clustering and cluster interpretation. Last: historical validation.

## How it can be improved

- A higher-quality data source (ready-made derived metrics, deeper history) — requires no interface change.
- Intra-sector normalization: metrics from different sectors can't be compared directly; solving this opens the engine to a broader universe.
- Richer metrics and clustering methods, but measured — each new metric carries noise risk, and the decision to add one must rest on the data.
- Deepening validation: testing the engine's past behavior more systematically.

## Future value

In the long run, this engine is the "quality" leg of an investment-thesis-generation tool. The goal is to combine financial discipline with technical/domain judgment in a single flow — surrendering to neither purely quantitative models nor pure intuition. It's also an engineering exercise in its own right: source-agnostic architecture, honest data handling, test discipline.

## Tech stack

Python · pandas · numpy · yfinance · scikit-learn (next) · pytest. A modular, layered, testable design.

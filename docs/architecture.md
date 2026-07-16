# Architecture

This document explains each stage of the graph, the reasoning behind the
design decisions, and where the system's boundaries are.

## Design goals

1. **Latency and cost proportional to ambiguity.** Clear cases (the vast
   majority in real traffic) are decided in microseconds by deterministic
   code. Only the ambiguous middle band pays for an LLM call.
2. **Explainability by construction.** Every decision carries rule IDs,
   score contributions, and an ordered audit trail. Nothing is a black box
   at decision time.
3. **Fail safe, not fail open or closed.** If the LLM is unavailable the
   system does not approve blindly (fail open) or block everything (fail
   closed); ambiguous cases degrade to human review.

## The graph

```
START -> ingest -> features -> rules -> scoring -> [triage] -> decide -> END
                                                       \-> llm_analyst -> decide
```

### ingest

Validates the raw payload against Pydantic models (`Transaction`,
`CustomerProfile`). Malformed input fails here, at the boundary, with a
clear validation error instead of a confusing failure mid-pipeline.

### features

Pure functions in `features.py` turn the transaction plus the customer's
behavioural baseline into signals:

| Signal | Meaning |
|---|---|
| `amount_zscore`, `amount_ratio` | How far the amount deviates from this customer's own baseline |
| `velocity_10m/1h/24h` | Prior transaction counts in sliding windows |
| `small_txn_burst_10m` | Sub-$5 charges in 10 minutes (card-testing precursor) |
| `travel_speed_kmh` | Implied speed since the last geo-located transaction (haversine distance / elapsed time) |
| `is_new_device`, `is_foreign`, `is_unusual_category`, `is_high_risk_category`, `is_night`, `is_online` | Context flags |

Everything is deterministic: same input, same output. This is what makes
decisions reproducible in a dispute.

### rules

Eight rules (R001-R008) encode known fraud patterns with three severities:

- **critical** (weight 1.0, hard block): blocklist hit, impossible travel
  (> 900 km/h, faster than any commercial flight), card-testing pattern.
- **high** (0.5): extreme amount anomaly, new device abroad, velocity spike.
- **medium** (0.25): high-risk merchant category with unusual amount,
  night-time online purchase from an unknown device.

The rule score is the capped sum of weights. A critical hit sets
`hard_block`, which routes straight to the decision node -- the LLM is
never consulted, because the block must be instant and citable.

### scoring

A weighted ensemble produces the anomaly score. Component weights:
amount 0.30, velocity 0.20, geo 0.20, device 0.15, context 0.15. The
amount component passes the |z-score| through a sigmoid shifted by 3, so
ordinary variation (under three standard deviations) contributes almost
nothing.

Anomaly and rule scores are fused with a noisy-OR:

```
combined = 1 - (1 - anomaly) * (1 - rule_score)
```

Noisy-OR is the textbook way to combine independent detectors of the same
event: evidence only compounds, either detector alone can raise the
total, and the result stays in [0, 1].

### triage (conditional edge)

- `hard_block` -> decide (BLOCK)
- combined < 0.30 -> decide (clear approve)
- combined >= 0.80 -> decide (clear block)
- otherwise -> llm_analyst (the gray zone)

### llm_analyst

The analyst receives the full case file -- transaction, baseline profile,
features, rule hits, score contributions -- and must return a structured
`LLMAssessment` (fraud probability, key factors, recommended action,
reasoning) via `with_structured_output`. Free-text answers are impossible
by construction, so downstream logic stays deterministic.

The default backend is Claude (`claude-sonnet-5`, override with
`FRAUD_AGENT_MODEL`). The node accepts any
`Callable[[dict], Optional[dict]]`, which is how tests inject fakes and
how a different model could be swapped in.

Returning `None` (no API key, or a failure handled by the caller) is a
first-class outcome: the graph proceeds on deterministic scores alone.

### decide

- Hard block: BLOCK at score 1.0 with the triggering rules cited first.
- With an LLM assessment: `final = 0.45 * combined + 0.55 * llm_probability`.
- Thresholds: APPROVE below 0.30, BLOCK at or above 0.70, REVIEW between.
- Guardrail: if the LLM recommends BLOCK but the fused score lands in the
  approve band, the decision escalates to REVIEW. The LLM can raise
  suspicion; it cannot single-handedly clear contradicting evidence.

The decision object carries the action, final score, risk band, ordered
reasons (rule details plus the LLM's key factors), the analyst's
narrative, and a `requires_human_review` flag.

## Threat model covered

- Card testing (micro-charge bursts before a real hit)
- Stolen card used far from the owner (impossible travel)
- Account takeover (new device, unusual amount, night-time online)
- Mule/blocklisted accounts
- Rapid-fire velocity abuse

## Known limitations

- The customer baseline travels with the request. Production would
  resolve it from a feature store inside the graph and cache it.
- The anomaly ensemble is hand-weighted. With labelled data, the natural
  upgrade is a trained model (gradient boosting or isolation forest)
  behind the same node interface.
- Thresholds are static. Real systems tune them continuously against
  confirmed-fraud feedback, and rules that stay static for a quarter
  measurably lose recall.
- Single-transaction scope. Cross-customer network signals (shared
  devices, merchant rings) need a graph database and are out of scope.

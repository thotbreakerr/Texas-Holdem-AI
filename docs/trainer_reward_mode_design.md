# Future Trainer Reward Mode Seam

Phase 2 keeps Deep CFR training on chip EV. Tournament winner-take-all is an
evaluation and checkpoint-selection concern only.

If a future eval-gated experiment explicitly approves a reward A/B, add a
trainer flag with this shape:

```text
--trainer-reward-mode chip_ev|wta_tournament
```

The default must remain `chip_ev`. The localized behavior switch is the
trainer leaf-value call that currently routes through `_aivat_leaf_value` with
`mode="chip_ev"`; the experimental branch would pass `mode="tournament"` with
winner-take-all payouts. That future change must keep schema-v2 checkpoint
loading explicit, or ship a separate schema migration/rejection path if it
changes target semantics.

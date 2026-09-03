# RESUME

1. Read STATE.md, HYPOTHESES.md, and the last entries of ledger.jsonl.
2. Check nyx for a running `run_zl010_analog.py --dataset ETTm2`; validate its JSON
   (`candidates/ZL-010/metrics_ETTm2.json`) before relaunching anything.
3. Execute the "Single next action" command in STATE.md.
4. Then apply the ONE permitted F2 refinement (temporal tokens + middle layer). If pretrained
   still loses to raw L2 pre-decoder on both datasets, KILL F2 and move to F3 (token kernel),
   which is the top-ranked open hypothesis.
5. Never tune on the six benchmark datasets; discovery stays on ETTh2/ETTm2 historical audit.

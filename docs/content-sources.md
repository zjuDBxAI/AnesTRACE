# Website content sources

The English and Chinese pages summarize [AnesTRACE, arXiv:2609.32740v1](https://arxiv.org/pdf/2609.32740v1), submitted 2026-09-26.
The clinical reading layout was revised on 2026-10-01.

| Website content | Paper locator | Boundary preserved |
| --- | --- | --- |
| Three complementary capability settings | Sections 3–4; Table 1 | Different input representation, information access and temporal scope; not one model-driven rollout |
| 1,817 perception questions; 493 anchors / 467 cases; 167 episodes / 517 turns / 124 cases | Section 4; Appendix B.1 | Counts use distinct units and are not added as a patient total |
| Three-turn donor-nephrectomy case, DeepSeek-V4-Pro | Appendix G; Figure 10, PDF page 33 | Model response summaries and evaluator judgments; not treatment instructions; later states are recorded rather than model-generated |
| 32.2 mIoU, 17.5% Safety Error, 74.3s p95, 82.7 L3 average | Tables 3–4, PDF page 7 | GPT-6-Astra; region overlap, response-level safety labels and timed turns are not patient outcomes |
| Quality scores and safety denominator | Section 5; Appendix B.3, PDF page 22 | Ordinal 0/1/2 scores normalized to 0–100; safety is separate; missing/failed safety judgments excluded |
| Timing and evidence-acquisition definitions | Appendix D.4, PDF page 27 | Model-request plus tool-execution time per turn; evidence acquisition is an exploratory matching proxy |
| Evaluator training | Section 5; Appendix C | Approx. 7k SFT examples and 4k DPO pairs; case overlap disclosed |
| Evaluator validation | Appendix C.3; Table 13, PDF pages 24–25 | 100 L2 responses; 117 L3 turns / 39 episodes; three anesthesiologists divide items, one annotation per item; validation-generator responses excluded from SFT/DPO |
| Limits of interpretation | Section 4; Conclusion | Retrospective selected cases; no established patient-outcome benefit, unseen-case generalization or deployment qualification |
| Submission and data access | Repository README and data/README.md | Locally runnable fixed weights, public Issues, no public data-request route |

Numerical model results in `leaderboard-data.js` were preserved. The page reorders
columns and adds sorting shortcuts; it does not create a composite rank across
quality, safety and latency. The agreement claims do not imply clinician consensus
on each annotation or safety certification.

Use `CONTEXT.md` in the parent manuscript workspace as the terminology authority.
Keep both language pages synchronized and check every substantive update against
the cited paper version.

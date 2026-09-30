# Data Availability

**AnesTRACE benchmark data are not currently distributed.** The public repository and website contain no patient records, clinical trajectories, gold answers, restricted imaging or waveform files, or source-derived benchmark questions. There is no public data-request process at present.

To request evaluation of a locally runnable, fixed-version model, follow [Model Submission](../README.md#model-submission) and contact **`[CONTACT_EMAIL]`**. This placeholder must be replaced before formal submissions open. The team runs the private test set; applicants do not receive test cases or reference answers. Do not attach weights, credentials, or clinical data to emails or public Issues.

Sources used in the study include [VitalDB](https://vitaldb.net/dataset/), [MOVER](https://mover.ics.uci.edu/download.html), [INSPIRE](https://physionet.org/content/inspire/), and [EchoNet-TEE](https://github.com/echonet/tee-view-classifier). Each has its own access and reuse terms. The AnesTRACE code license grants no rights to these datasets and does not authorize redistribution.

For authorized internal testing, pass an explicit JSONL path or place private files under `data/private/`. This directory is Git-ignored. Files in `examples/synthetic/` are artificial interface checks, not benchmark data.

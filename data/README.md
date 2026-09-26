# Dataset Access and Placement

## Open Research Release and Data Access

This repository provides the AnesTRACE research code and documents how to
access the benchmark data. Clinical data files are not included in the
public GitHub repository. Access to source-derived benchmark components is
subject to the licenses, data use agreements, and permissions of their
underlying datasets: VitalDB, MOVER, INSPIRE, and EchoNet-TEE.

The software license does not grant access to, or override the terms of,
third-party clinical data. Openly available code does not imply unrestricted
redistribution of patient-level records, waveforms, images, videos, or
benchmark questions and answers containing source-derived clinical content.

## Step 1: Obtain Source-Dataset Access

For access to the complete benchmark, satisfy the applicable requirements
for all four sources. For a source-specific subset, satisfy the requirements
for every source represented in that subset. Follow the official terms for
the source version used in the requested AnesTRACE release.

| Source      | Official access route                                        | Required action                                              |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| VitalDB     | [VitalDB Open Dataset](https://vitaldb.net/dataset/)         | Read and accept the official data use agreement and comply with its license. The public download route does not require the same credentialing process as INSPIRE; an acknowledgement of the applicable terms is sufficient for the AnesTRACE source-access check unless the provider requires additional authorization. |
| MOVER       | [MOVER download request](https://mover.ics.uci.edu/download.html) | Complete and sign the official access form, verify the email address as instructed, and obtain the download credentials. Request access to the relevant records and waveform components. Retain the access confirmation. |
| INSPIRE     | [INSPIRE on PhysioNet](https://physionet.org/content/inspire/) | Create a PhysioNet account, obtain credentialed-user status, complete the training required by the selected release, submit the training documentation, and sign the dataset-specific data use agreement. Retain evidence that access is enabled. |
| EchoNet-TEE | [Official project](https://github.com/echonet/tee-view-classifier) and [data availability statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC10761863/) | Request access to the TEE data from the corresponding author and obtain the required Stanford AIMI permission, following the provider's current instructions. The publication lists `ksteffner@stanford.edu` and `aimicenter@stanford.edu` as contacts. Access to the code alone is not authorization to use or redistribute the imaging data. |

EchoNet-TEE and EchoNet-Dynamic are different datasets. Approval for another
EchoNet dataset does not automatically authorize access to EchoNet-TEE.
The upstream providers determine eligibility, required training, and
permitted use; their current instructions take precedence over this summary.

## Step 2: Submit an AnesTRACE Data Request

**Request channel: [TO BE ADDED: AnesTRACE data-access email or application form].**

After completing the applicable source-access steps, submit:

- Your name, affiliation, and institutional email address.
- A brief description of the research purpose and intended use.
- The requested AnesTRACE release, levels, language versions, and source subsets.
- Evidence of access approval for restricted sources, such as a redacted
  approval email or access-status confirmation. For VitalDB, acknowledge
  acceptance of the applicable license and data use agreement.
- Confirmation that every intended data user has the required individual
  authorization, where required by the source provider.
- Confirmation that data will be stored securely and used in accordance
  with all applicable source agreements and institutional requirements.

Do not submit passwords, API keys, download credentials, private download
links, or patient records. Do not post access documentation or clinical
content in public GitHub issues or pull requests.

## Step 3: Access Verification and Authorized Delivery

The AnesTRACE maintainers review the requested scope and source-access
documentation, and check whether the requested artifacts may be shared
under the applicable upstream terms. Additional provider permission may
be required before source-derived artifacts can be delivered.

Source-access approval is a prerequisite where applicable, not permission
for the AnesTRACE team to redistribute the source data. In particular,
the [MOVER agreement](https://mover.ics.uci.edu/download.html) prohibits
redistribution, and the INSPIRE agreement restricts disclosure to others.
These restrictions must be respected even when the applicant has their
own source-dataset access.

Following verification, only artifacts whose delivery is permitted will
be provided through the designated access channel. If redistribution of a
requested component is not permitted, it will not be transferred;
the underlying data must be obtained directly from its provider. Any
source-derived benchmark release remains subject to the relevant
permissions and agreements. AnesTRACE access does not transfer ownership
of the underlying data or authorize onward sharing.

## Conditions of Use

- Follow the licenses and agreements for each source represented in the
  requested data, including restrictions on research or commercial use.
- Do not attempt to identify patients or institutions from de-identified
  records, or link records for re-identification.
- Do not redistribute restricted data, patient-level benchmark files,
  source-derived media, or private access links without the necessary
  provider permissions.
- Do not upload restricted clinical content to third-party APIs or public
  services unless permitted by the source agreements and applicable
  institutional requirements.
- Cite AnesTRACE and every source dataset used, including the relevant
  dataset versions and publications.
- Use the benchmark for research evaluation, not as a clinical decision
  system or a substitute for professional medical judgment.
- Report suspected identifying information or unauthorized disclosure to
  the relevant source provider and the AnesTRACE maintainers, following
  any reporting deadlines in the applicable agreements.

## Local Dataset Placement

After obtaining authorized copies, place them in this directory, set
`ANESTRACE_DATA_DIR`, or pass an explicit JSONL path to `run_inference.py`.

The default filenames are:

- `Level_two_B5_v2_zh.jsonl`
- `Level_two_B5_v2_en.jsonl`

Each record must contain non-empty `qa_id`, `sample_id`, and
`patient_information` fields, with:

- `task_name`: `B5_complete_plan`
- `answer_type`: `open_ended`
- `split`: `train`, `validation`, or `test`

Other fields may be present, but the model receives only
`patient_information`. Dataset files are ignored by Git to reduce the risk
of committing protected clinical content.

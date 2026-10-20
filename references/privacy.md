# Privacy and Publication Rules

Visual evidence can contain identifiable and sensitive information. Apply data minimization before running the pipeline and again before sharing outputs.

## Public repositories

Never commit:

- recordings of real people without documented consent and a lawful basis;
- biometric galleries, identity mappings, or face embeddings;
- private case notes, interview transcripts, dates of birth, addresses, or account identifiers;
- raw JSONL or annotated frames produced from private media;
- model weights with incompatible licenses.

Use synthetic fixtures for tests. Keep real media outside the repository and add a local ignore rule for every output directory.

## Runtime handling

1. Process local media offline; the script makes no network requests.
2. Store only the fields needed for the stated task.
3. Set a retention period and delete intermediate annotated frames when they are no longer needed.
4. Restrict access to the input and output directories.
5. Do not add identity recognition or emotion labels to this pipeline without a new necessity, impact, and legal review.

## Sharing evidence

When an output must be shared, remove exact timestamps, file names, and frames that reveal bystanders. Provide the JSONL aggregate and a short methods note instead of raw media when the recipient only needs quality statistics.

## Limits of consent

Consent does not automatically make secondary use, biometric profiling, or indefinite retention acceptable. Follow the applicable jurisdiction and organizational policy.

# Source and license notices

This is a mixed-source research artifact. Do not describe the entire data bundle as Apache-2.0 or as unrestricted. Dataset license notices do not, by themselves, settle every question about a resulting model's license or a particular redistribution/use.

- Base model: Qwen/Qwen3.5-0.8B, pinned revision `2fc06364715b967f1860aea9cf38778875588b17`, Apache-2.0. The base weights are downloaded separately. Original license: `licenses/QWEN_APACHE_LICENSE`. Source: https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17
- Exact unmodified Clef reference schema head: Cloudflare, vendored in Unsloth at commit `0b1291bfd7c147e2edf82c085b8412c2630be4fe`, Apache-2.0. Source and original license are in `upstream/`. Source: https://github.com/unslothai/unsloth/blob/0b1291bfd7c147e2edf82c085b8412c2630be4fe/unsloth/_vendor/clef/joint_schema_model.py
- Original Typed Decisions public data: LocalLLaMA/typed-decisions revision `6a7927fa3c8ac114e29f5300eacb86f6f37a6447`, dataset card declares Apache-2.0. Teacher-generated labels measure agreement with those teachers. Card: `licenses/TYPED_DECISIONS_DATASET_CARD.md`. Source: https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/6a7927fa3c8ac114e29f5300eacb86f6f37a6447
- Amazon MASSIVE 1.1: CC BY 4.0. Retain the original Amazon attribution, citation and localization notices under `licenses/public_bilingual/`.
- SNLI 1.0 and SNLI-TR 1.1: CC BY-SA 4.0 as stated by the source projects. Retain the applicable attribution and ShareAlike notices for redistributed data adaptations. Turkish SNLI was machine-translated by the original NLI-TR authors; this project did not independently validate those translations.

Full data authors, official URLs, transformations, archive versions and license notices are in `licenses/BILINGUAL_SOURCES_AND_LICENSES.md` and the original study source manifest. Fixed bilingual schemas and CPU training/inference glue were authored for this experiment. Only the Apache-licensed Clef reference runtime was imported; the Unsloth AGPL sizing recipe consulted during implementation research is not included or imported as runtime code.

No external teacher service was called, no private user corpus was used, and no model or data was published to an external model host during this experiment. The separately authored DecisionMix dataset was not used for these weights or evaluation.

Publication layout note: this repository contains source attribution documents, not the original corpus records. See `LICENSE_STATUS.md` for trained-weight license status.

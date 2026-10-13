> Historical dataset-source notices. Corpus records and raw source archives are not included in this upload repository.

# Sources, attribution and licensing

This is a mixed-source dataset conversion. Preserve the source notices and per-record licenses when redistributing the data. Do not describe the entire data pack as Apache-2.0, unrestricted, human-reviewed by this project, or newly collected native Turkish text. Dataset notices do not by themselves settle every question about a resulting model's license.

## Used for training: Amazon MASSIVE 1.1

- Authors: Jack FitzGerald, Christopher Hench, Charith Peris and collaborators; Amazon.com, Inc. or its affiliates.
- Official project: https://github.com/alexa/massive
- Official versioned release: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
- Paper: https://arxiv.org/abs/2204.08582
- Source of English utterances: SLURP, Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski and Verena Rieser (2020), https://aclanthology.org/2020.emnlp-main.588/ and https://github.com/pswietojanski/slurp
- Data license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Original release license and notices are copied in `licenses/MASSIVE_LICENSE`, `MASSIVE_NOTICE.md` and `MASSIVE_CITATION.md`.
- The paper identifies professional translators as the source of localization. This work relies on the original corpus labels and source quality judgments; it did not commission or independently review the localizations.
- Transformations: fixed intent-schema descriptions authored in TR/EN; source intents mapped to short output keys; conservative exact-text/slot-template family grouping; deterministic domain-balanced sampling; selected metadata retained separately from model inputs.

## Used for training: SNLI 1.0 and SNLI-TR 1.1

- SNLI authors: Samuel R. Bowman, Gabor Angeli, Christopher Potts and Christopher D. Manning (2015).
- Official project and license statement: https://nlp.stanford.edu/projects/snli/
- Official archive: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
- Paper: https://aclanthology.org/D15-1075/
- Underlying Flickr30k captions: Peter Young, Alice Lai, Micah Hodosh and Julia Hockenmaier (2014), https://aclanthology.org/Q14-1006/ . SNLI notes additional pilot captions from VisualGenome.
- SNLI-TR authors: Emrah Budur, Rıza Özçelik, Tunga Güngör and Christopher Potts (2020).
- Official project and license statement: https://github.com/boun-tabi/NLI-TR
- Official archive: https://nli-tr.s3.eu-central-1.amazonaws.com/snli_tr_1.1.zip
- Paper: https://aclanthology.org/2020.emnlp-main.662/
- Data license for both: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**, as stated by the original projects. Retain attribution and applicable ShareAlike obligations for redistributed adaptations. Original READMEs and a license notice with the CC BY-SA URI are included in `licenses/`.
- Turkish text was machine-translated using Amazon Translate by the original NLI-TR authors. Some original examples carry translation-review metadata; that does not mean every selected example is human-reviewed. This work makes no such claim.
- Transformations: EN/TR pairing by original `pairID`; grouping all captions and translations by original source image; strict later-partition-family exclusion; English training-label agreement filter; deterministic balanced sampling; bilingual fixed NLI question descriptions; source labels unchanged.

## Evaluated as future options, not included

- **XNLI**: valuable Turkish/English cross-lingual NLI evaluation, but the official repository license is CC BY-NC 4.0. It is not part of this training pack. Sources: https://github.com/facebookresearch/XNLI and https://github.com/facebookresearch/XNLI/blob/main/LICENSE . A non-commercial-only source should not silently become part of an unrestricted release.
- **BOUN Turkish sentiment (BounTi)**: official train/dev/test labels are available, but the original project shares tweet IDs rather than tweet text because of source-platform restrictions. It requires a separate authorized text-retrieval path and was not used. Source: https://github.com/boun-tabi/BounTi-Turkish-Sentiment-Analysis .
- **FSMTSA sentiment**: the project advertises a CC0 repository license while its usage guidance says academic/research use and requires permission for commercial use. It also mixes LLM-generated material. This unresolved scope/provenance combination is unsuitable for silently adding to this release. Source: https://github.com/kevserbusrayildirim/FSMTSAD .
- **Other scraped Turkish review/sentiment compilations**: not incorporated without verifying original data rights, deduplication lineage and source-domain independence. A repository software license alone is not evidence that all underlying texts carry the same terms.

## Reproducible identity

`SOURCE_MANIFEST.json` records the official archive URL, version, exact byte count and SHA-256 for every included source. Archive bytes, not mutable repository heads or a dataset card's title, determine this preparation version. `data/manifest.json` separately freezes the converted partitions. No external dataset Python loader was executed.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.7.4] - 2025-11-05

### Added

- Normalize ligatures in word_segmenatation: "scientiﬁc" "ﬁelds" "ﬁnal".
- Works with upper case input in word_segmentation.
- Retains/preserves letter case in word_segmentation.
- Applies spelling correction during word_segmentation to allow noisy text with spelling mistakes.
- Keep punctuation or apostrophe adjacent to the previous word in word_segmentation.
- Ported more comments from C# to Rust.

## [6.7.3] - 2025-11-05

### Added

- First release of the official Rust implementation of SymSpell v6.7.3

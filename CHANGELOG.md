# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.8.2] - 2025-11-27

### Changed

- `lookup_compound` now also with new parameter `term_length_threshold`.

## [6.8.1] - 2025-11-27

### Changed

- README.md included in documentation tests (doctest) to ensure that code examples are always correct and up-to-date.

## [6.8.0] - 2025-11-25

### Added

- `create_dictionary_entry` exposed as public method, for incremental dictionary update.
- `save_dictionary` to write the dictionary to a CSV file.
- `lookup()` with new parameter `max_results` to limit the number of suggestion returned.
- `SymSpell::new()` and `lookup()` with new parameter `term_length_threshold`: A vector of term length thresholds for each maximum edit distance.  
  Allowing higher maximum edit distances for longer terms, with minimal additional latency and memory consumption.
  - None: max_dictionary_edit_distance for all terms lengths
  - Some([4].into()): max_dictionary_edit_distance for all terms lengths >= 4,
  - Some([2,8].into()): max_dictionary_edit_distance for all terms lengths >=2, max_dictionary_edit_distance +1 for all terms for lengths>=8

### Fixed

- In `create_dictionary_entry` words weren't added as stub for count < count_threshold, which is required for correct incremental directory entry creation and counting.

### Changed 

- Refactoring.
- hash64 replaced with hash32.

## [6.7.8] - 2025-11-15

### Fixed

- Sort order for suggestions fixed.

## [6.7.7] - 2025-11-14

### Added

- lookup_compound() now allows input in upper/lower case.
- lookup_compound() with new `preserve_case` parameter : Whether to preserve the letter case from input to suggestion.
- transfer_case doesn't transfer lower case of whitespace in source to non-whitespace char in target.

## [6.7.6] - 2025-11-14

### Added

- lookup() now allows input in upper/lower case.
- lookup() with new `preserve_case` parameter : Whether to preserve the letter case from input to suggestion.

## [6.7.5] - 2025-11-14

### Added

- word_segmentation now supports Chinese word segmentation.
- Chinese dictionary added frequency_dictionary_zh_cn_349_045.txt

## [6.7.4] - 2025-11-12

### Added

- Normalize ligatures in word_segmenatation: "scientiﬁc" "ﬁelds" "ﬁnal".
- Works with upper case input in word_segmentation.
- Retains/preserves letter case in word_segmentation.
- Applies spelling correction during word_segmentation to allow noisy text with spelling mistakes.
- Keep punctuation or apostrophe adjacent to the previous word in word_segmentation.
- Ported more comments from C# to Rust.
- WordSegmentation now removes hyphens prior to word segmentation (as they might be caused by syllabification).
- American English word forms added to frequency_dictionary_en_82_765.txt in addition to British English e.g. *favourable -> favorable*.
- More common contractions added to frequency_dictionary_en_82_765.txt e.g. *hasn't*.

## [6.7.3] - 2025-11-05

### Added

- First release of the official Rust implementation of SymSpell v6.7.3

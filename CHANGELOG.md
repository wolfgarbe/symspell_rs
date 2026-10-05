# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.0.1] - 2026-10-04

### Added

- New basic benchmark added 
  - Lookup latency experiments (162 in total): the 30k, 82k and 500k English frequency dictionaries × `prefix_length` 5, 6, 7 × maximum edit distance 1, 2, 3 × `Verbosity` Top, Closest, All × {this version, [symspell_rs 6.8.4](https://crates.io/crates/symspell_rs/6.8.4)}.
  - Load dictionary time experiments (27 in total): the 30k, 82k and 500k English frequency dictionaries × `prefix_length` 5, 6, 7 × maximum edit distance 1, 2, 3
  - Damerau-Levenshtein OSA latency experiments (6 in total): maximum edit distance 1, 2, 3 x bit-parallel OSA, multi-word block bit-parallel OSA
  - **Basic benchmark**: `cargo bench --bench basic   --features gxhash` 
  - **Verbose benchmark**, including RAM consumption, based on [divan](https://github.com/nvzqz/divan): `cargo bench --bench verbose  --features gxhash`
  - See [detailed benchmark results](benches\results\RESULTS.md).

## [7.0.0] - 2026-10-02

### Added
- `lookup` benchmark (`benchmark/lookup.rs`) based on [divan](https://github.com/nvzqz/divan). Run it with `cargo bench`.
- Queries: the first term of each line of `benchmark/test_data/noisy_query_en_1000.txt` (1000 misspelled terms).
- Experiments (162 in total): the 30k, 82k and 500k English frequency dictionaries × `prefix_length` 5, 6, 7 × maximum edit distance 1, 2, 3 × `Verbosity` Top, Closest, All × {this version, [symspell_rs 6.8.4](https://crates.io/crates/symspell_rs/6.8.4)}. The dictionary is rebuilt for each maximum edit distance (maximum dictionary edit distance = maximum edit distance). 6.8.4 is pulled in as the dev-dependency `symspell_old`.
- Measured per experiment: the time of one pass over all queries (reported by divan as items/s), the average latency per `lookup()`, and the peak heap allocation during lookups (via a tracking global allocator).
- Measured per dictionary build: build time, resident memory and peak memory during the build, for both versions.
- After each 6.8.4 run, the benchmark prints the latency speedup and the peak-allocation ratio of this version relative to 6.8.4.
- The average latency and memory figures are printed to stderr next to the divan output.
- Run a subset with a divan filter, e.g. `cargo bench -- 82_765`.

### Performance: 3x faster lookup, 40% less memory consumption.

- **Faster lookups in every experiment.** All 81 lookup experiments are faster, by a geometric mean of **2.8×** (range 1.5× to 5.9×).
  - `Verbosity::Top`: 3.0× on average (1.8× to 5.9×)
  - `Verbosity::Closest`: 2.9× on average (1.5× to 4.2×)
  - `Verbosity::All`: 2.7× on average (1.6× to 4.8×)
  - max edit distance 1: 2.4× on average (1.5× to 3.7×)
  - max edit distance 2: 2.8× on average (2.1× to 5.2×)
  - max edit distance 3: 3.4× on average (1.8× to 5.9×)
  - 30,000-word dictionary: 2.9× on average (1.5× to 4.8×)
  - 82,765-word dictionary: 3.0× on average (1.8× to 5.9×)
  - 500,000-word dictionary: 2.6× on average (1.6× to 3.5×)
- **Much lower memory use during lookups.** Peak heap allocation per lookup is on average only **31%** of v6.8.4 (best case 17%, worst case 77%).
  - `Verbosity::Top`: 18% of v6.8.4 on average (17% to 18%)
  - `Verbosity::Closest`: 32% of v6.8.4 on average (18% to 73%)
  - `Verbosity::All`: 52% of v6.8.4 on average (33% to 77%)
- **Roughly 40% less memory for the dictionary.** Resident memory after the build is on average **62%** of v6.8.4 (best case 48%, worst case 83%). Peak memory during the build is on average **64%** of v6.8.4 (48% to 88%).
  - Largest example, 500k dictionary, prefix 7, edit distance 3: resident memory 874 MiB → 436 MiB, peak 874 MiB → 436 MiB, build time 14.20 s → 12.60 s.
- See [detailed benchmark results](benches\results\RESULTS.md).

### Changed
- Internal storage: `words` now maps terms to ids, and counts live in a new term table. Serialized dictionaries from previous versions (`serde` feature) are not compatible.
- The derived `PartialEq` on `SymSpell` compares term ids, so dictionaries with the same content but a different insertion order compare unequal.
- ASCII input now takes an allocation-free fast path: stack-based candidates, borrowed hits, and `Suggestion` strings created only for the returned results (after sort and `max_results`). Non-ASCII input uses a char-based path that shares the same verification code.
- Delete buckets store compact 8-byte entries (term id, length, ASCII flag) that refer to a shared term table, instead of a heap copy of the term per delete. This lowers memory use and removes pointer chasing.
- The `deletes` map uses a lightweight hasher, since its keys are already 32-bit hashes.

## [6.9.1] - 2026-10-01

### Improved

- Optimized `damerau_levenshtein_osa_fallback` by using the **multi-word block version** of the **bit-parallel algorithm ([Hyyrö 2003](https://www.sciencedirect.com/science/article/pii/S157086670400053X/pdf))**, to cover any length, making it **6.9x faster** than `strsim.osa_distance` for terms > 64 chars.

## [6.9.0] - 2026-09-29

### Improved

- Optimized `damerau_levenshtein_osa` by using **bit-parallel OSA ([Hyyrö 2003](https://www.sciencedirect.com/science/article/pii/S157086670400053X/pdf))**, making it **10.9x faster** than `strsim.osa_distance` and boosting SymSpell lookup speeds by 20%.
- Exposed `damerau_levenshtein_osa` as a public method for standalone use.

## [6.8.4] - 2026-09-28

### Changed

- Explicit activation of new `gxhash`-feature for high-performance hashing via `gxhash`, both for `x86_64` and `aarch64`. Enable the `gxhash`-feature:
  1. If compiling for x86_64 AND explicitly targeted AES/SSE2: `#[cfg(all(target_arch = "x86_64", target_feature = "aes", target_feature = "sse2"))]` or
  2. If compiling for ARM64 AND explicitly targeted AES/NEON`#[cfg(all(target_arch = "aarch64", target_feature = "aes", target_feature= "neon"))]`
  3. Otherwise fallback to `ahash`.

## [6.8.3] - 2025-12-05

### Fixed

- Gracefully handle empty input strings for word_segmentation. Fixes #1 .

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

// SymSpell: 1 million times faster through Symmetric Delete spelling correction algorithm
//
// The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and dictionary lookup
// for a given Damerau-Levenshtein distance. It is six orders of magnitude faster and language independent.
// Opposite to other algorithms only deletes are required, no transposes + replaces + inserts.
// Transposes + replaces + inserts of the input term are transformed into deletes of the dictionary term.
// Replaces and inserts are expensive and language dependent: e.g. Chinese has 70,000 Unicode Han characters!
//
// SymSpell supports compound splitting / decompounding of multi-word input strings with three cases:
// 1. mistakenly inserted space into a correct word led to two incorrect terms
// 2. mistakenly omitted space between two correct words led to one incorrect combined term
// 3. multiple independent input terms with/without spelling errors

// Copyright (C) 2025 Wolf Garbe
// Version: 6.7.3
// Author: Wolf Garbe wolf.garbe@seekstorm.com
// Maintainer: Wolf Garbe wolf.garbe@seekstorm.com
// URL: https://github.com/wolfgarbe/symspell
// Description: https://seekstorm.com/blog/1000x-spelling-correction/
//
// MIT License
// Copyright (c) 2025 Wolf Garbe
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// https://opensource.org/licenses/MIT

use ahash::{AHashMap, AHashSet};
use itertools::Itertools;
use std::cmp;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

#[cfg(not(all(target_feature = "aes", target_feature = "sse2")))]
use ahash::RandomState;
#[cfg(not(all(target_feature = "aes", target_feature = "sse2")))]
use std::sync::LazyLock;

#[cfg(not(all(target_feature = "aes", target_feature = "sse2")))]
pub static HASHER_32: LazyLock<RandomState> =
    LazyLock::new(|| RandomState::with_seeds(805272099, 242851902, 646123436, 591410655));

// stable hash, faster, but not available on all platforms
// https://github.com/tkaitchuck/aHash
#[inline]
#[cfg(all(target_feature = "aes", target_feature = "sse2"))]
pub(crate) fn hash32(term_bytes: &[u8]) -> u32 {
    use gxhash::gxhash32;

    gxhash32(term_bytes, 1234)
}

// unstable hash, slower, but available on all platforms
// https://github.com/ogxd/gxhash
#[inline]
#[cfg(not(all(target_feature = "aes", target_feature = "sse2")))]
pub(crate) fn hash32(term_bytes: &[u8]) -> u32 {
    HASHER_32.hash_one(term_bytes) as u32
}

use std::{cmp::min, mem};

use smallvec::SmallVec;
use smallvec::smallvec;

const VEC_SIZE: usize = 16; //32
pub type FastVec<T> = SmallVec<[T; VEC_SIZE]>;

/// Damerau-Levenshtein edit distance, like Levenshtein but allows for adjacent transpositions.
/// Optimal string alignment version (OSA): each substring can only be edited once.
/// E.g., "CA" to "ABC" has an edit distance of 2 by for Damerau-Levenshtein, but a distance of 3 when using the optimal string alignment algorithm.
/// Returns the edit distance, >= 0 representing the number of edits required to transform one string to the other,
/// or -1 if the distance is greater than the specified max_distance.
/// https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance
pub fn damerau_levenshtein_osa(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    let b_len = b.chars().count();

    //the edit distance can't be less than the difference of the lengths of the strings.
    //if a.chars().count().abs_diff(b_len)> max_distance {return -1;}

    // 0..=b_len behaves like 0..b_len.saturating_add(1) which could be a different size
    // this leads to significantly worse code gen when swapping the vectors below
    let mut prev_two_distances: FastVec<usize> = (0..b_len + 1).collect();
    let mut prev_distances: FastVec<usize> = (0..b_len + 1).collect();
    let mut curr_distances: FastVec<usize> = smallvec![0; b_len + 1];

    let mut prev_a_char = char::MAX;
    let mut prev_b_char = char::MAX;

    for (i, a_char) in a.chars().enumerate() {
        curr_distances[0] = i + 1;

        for (j, b_char) in b.chars().enumerate() {
            let cost = usize::from(a_char != b_char);
            curr_distances[j + 1] = min(
                curr_distances[j] + 1,
                min(prev_distances[j + 1] + 1, prev_distances[j] + cost),
            );
            if i > 0 && j > 0 && a_char != b_char && a_char == prev_b_char && b_char == prev_a_char
            {
                curr_distances[j + 1] = min(curr_distances[j + 1], prev_two_distances[j - 1] + 1);
            }

            prev_b_char = b_char;
        }

        mem::swap(&mut prev_two_distances, &mut prev_distances);
        mem::swap(&mut prev_distances, &mut curr_distances);
        prev_a_char = a_char;
    }

    if prev_distances[b_len] <= max_distance {
        Some(prev_distances[b_len])
    } else {
        None
    }
}

/// Normalize ligatures: "scientiﬁc" "ﬁelds" "ﬁnal"
pub fn unicode_normalization_form_kc(input: &str) -> String {
    input
        .nfkc() // Apply Unicode Normalization Form KC
        .collect::<String>() // Collect normalized chars into a String
}

/// Transfer the letter case char-wise from source to target string.
pub fn transfer_case(source: &str, target: &str) -> String {
    // source = "HeLLo WoRLd!";
    // target = "rustacean community!";
    // result = "RuSTacEaN community!";

    //shortcut: if input and suggestion are identical or case-insensitive identical, no transfer required
    if source == target || source.eq_ignore_ascii_case(target) {
        return source.to_string();
    }

    let mut result = String::new();

    // iterate over both strings using zip_longest from itertools
    use itertools::EitherOrBoth;
    use itertools::Itertools;
    let mut last_upper = false;

    for pair in source.chars().zip_longest(target.chars()) {
        match pair {
            // both characters exist
            EitherOrBoth::Both(s, t) => {
                if s.is_uppercase() {
                    //don't memorize letter case for first character
                    if !result.is_empty() {
                        last_upper = true;
                    }
                    result.push_str(&t.to_string().to_uppercase());
                }
                /*
                // we don't need to lowercase, because dictionary words are already lowercased
                else if s.is_lowercase() {
                    result.push_str(&t.to_string().to_lowercase());
                    result.push(t);
                }
                */
                // don't transfer lower case of whitespace in source to non-whitespace char in target
                else if s.is_whitespace() && !t.is_whitespace() && last_upper {
                    result.push_str(&t.to_string().to_uppercase());
                } else {
                    last_upper = false;
                    result.push(t);
                }
            }
            // only the source has characters left — just append them as-is
            EitherOrBoth::Left(_) => (),
            // only the target has characters left — append unchanged
            EitherOrBoth::Right(t) => {
                //use memorizeed case from last char for exceeding chars
                if last_upper {
                    result.push_str(&t.to_string().to_uppercase());
                } else {
                    result.push(t);
                }
            }
        }
    }
    result
}

/// Parse a string into words, splitting at non-alphanumeric characters, except for underscore and apostrophes.
pub fn parse_words(text: &str) -> Vec<String> {
    let mut non_unique_terms_line: Vec<String> = Vec::with_capacity(text.len() << 3);
    let text_normalized = text.to_lowercase();
    let mut start = false;
    let mut start_pos = 0;

    for char in text_normalized.char_indices() {
        start = match char.1 {
            //start of term
            token if token.is_alphanumeric() => {
                //token if regex_syntax::is_word_character(token) => {
                if !start {
                    start_pos = char.0;
                }
                true
            }

            // allows underscore and apostrophes as part of the word
            '_' | '\'' | '’' => true,

            //end of term
            _ => {
                if start {
                    non_unique_terms_line.push(text_normalized[start_pos..char.0].to_string());
                }
                false
            }
        };
    }

    if start {
        non_unique_terms_line.push(text_normalized[start_pos..text_normalized.len()].to_string());
    }

    non_unique_terms_line
}

fn len(s: &str) -> usize {
    s.chars().count()
}

fn remove(s: &str, index: usize) -> String {
    s.chars()
        .enumerate()
        .filter(|(ii, _)| ii != &index)
        .map(|(_, ch)| ch)
        .collect()
}

fn slice(s: &str, start: usize, end: usize) -> String {
    s.chars().skip(start).take(end - start).collect()
}

fn suffix(s: &str, start: usize) -> String {
    s.chars().skip(start).collect::<String>()
}

fn at(s: &str, i: isize) -> Option<char> {
    if i < 0 || i >= s.len() as isize {
        return None;
    }

    s.chars().nth(i as usize)
}

#[derive(Debug, Clone)]
pub struct Composition {
    // the word segmented and spelling corrected string,
    pub segmented_string: String,
    // the Edit distance sum between input string and corrected string,
    pub distance_sum: usize,
    // the Sum of word occurence probabilities in log scale (a measure of how common and probable the corrected segmentation is).
    pub prob_log_sum: f64,
}

impl Composition {
    pub fn empty() -> Self {
        Self {
            segmented_string: "".to_string(),
            distance_sum: 0,
            prob_log_sum: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Suggested correct spelling for a given input word.
pub struct Suggestion {
    /// The suggested correctly spelled word.
    pub term: String,
    /// Edit distance between searched for word and suggestion.
    pub distance: usize,
    /// Frequency of suggestion in the dictionary (a measure of how common the word is).
    pub count: usize,
}

impl Suggestion {
    fn empty() -> Suggestion {
        Suggestion {
            term: "".to_string(),
            distance: 0,
            count: 0,
        }
    }

    fn new(term: impl Into<String>, distance: usize, count: usize) -> Suggestion {
        Suggestion {
            term: term.into(),
            distance,
            count,
        }
    }
}

// Order by distance ascending, then by frequency count descending
impl Ord for Suggestion {
    fn cmp(&self, other: &Suggestion) -> Ordering {
        let distance_cmp = self.distance.cmp(&other.distance);
        if distance_cmp == Ordering::Equal {
            return other.count.cmp(&self.count);
        }
        distance_cmp
    }
}

impl PartialOrd for Suggestion {
    fn partial_cmp(&self, other: &Suggestion) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Suggestion {
    fn eq(&self, other: &Suggestion) -> bool {
        self.distance == other.distance && self.count == other.count
    }
}
impl Eq for Suggestion {}

#[derive(Eq, PartialEq, Debug)]
/// Controls the closeness/quantity of returned spelling suggestions.
pub enum Verbosity {
    /// Top suggestion with the highest term frequency of the suggestions of smallest edit distance found.
    Top,
    /// All suggestions of smallest edit distance found, suggestions ordered by term frequency.
    Closest,
    /// All suggestions within maxEditDistance, suggestions ordered by edit distance, then by term frequency (slower, no early termination)
    All,
}

#[derive(PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// SymSpell spell checker and corrector.
pub struct SymSpell {
    /// Maximum edit distance for dictionary precalculation.
    max_dictionary_edit_distance: usize,
    /// Term length thresholds for each edit distance
    ///   None: max_dictionary_edit_distance for all terms lengths
    ///   Some([4].into()): max_dictionary_edit_distance for all terms lengths >= 4,
    ///   Some([2,8].into()): max_dictionary_edit_distance for all terms lengths >=2, max_dictionary_edit_distance +1 for all terms for lengths>=8
    ///   ⚠️ The resulting maximum edit distance defined in lookup() for each term length (max_edit_distance + term_length_threshold) must be
    ///   <= the corresponding maximum edit distance defined for each term length in SymSpell::new() used for creation of dictionary structures.
    term_length_threshold: Option<Vec<usize>>,
    /// The length of word prefixes, from which deletes are generated. (5..7).
    prefix_length: usize,
    /// The minimum frequency count for dictionary words to be considered a valid for spelling correction.
    count_threshold: usize,
    /// Number of all words in the corpus used to generate the frequency dictionary
    /// this is used to calculate the word occurrence probability p from word counts c : p=c/N
    /// N equals the sum of all counts c in the dictionary only if the dictionary is complete, but not if the dictionary is truncated or filtered
    corpus_word_count: usize,
    /// Maximum dictionary term length
    max_dictionary_term_length: usize,
    /// Dictionary that contains a mapping of lists of suggested correction words to the hashCodes
    /// of the original words and the deletes derived from them. Collisions of hashCodes is tolerated,
    /// because suggestions are ultimately verified via an edit distance function.
    /// A list of suggestions might have a single suggestion, or multiple suggestions.
    deletes: AHashMap<u32, Vec<Box<str>>>,
    // Dictionary of unique correct spelling words, and the frequency count for each word.
    words: AHashMap<Box<str>, usize>,
    /// Bigrams optionally used for improved correction quality in lookup_coompound
    bigrams: AHashMap<Box<str>, usize>,
    /// Minimum bigram count in the bigram dictionary
    bigram_min_count: usize,
}

// inexpensive and language independent: only deletes, no transposes + replaces + inserts
// replaces and inserts are expensive and language dependent (Chinese has 70,000 Unicode Han characters)
fn edits(
    word: &str,
    edit_distance: usize,
    max_term_edit_distance: usize,
    delete_words: &mut AHashSet<String>,
) {
    let edit_distance = edit_distance + 1;
    let word_len = len(word);

    if word_len > 1 {
        for i in 0..word_len {
            let delete = remove(word, i);

            if !delete_words.contains(&delete) {
                delete_words.insert(delete.clone());

                //max_term_edit_distance dependent on term_length_threshold
                if edit_distance < max_term_edit_distance {
                    edits(&delete, edit_distance, max_term_edit_distance, delete_words);
                }
            }
        }
    }
}

impl SymSpell {
    /// Creates a new SymSpell instance.
    ///  
    /// # Arguments
    ///
    /// * `max_dictionary_edit_distance` - Maximum edit distance for dictionary precalculation.
    /// * `term_length_threshold` - Term length thresholds for each edit distance.
    ///   None: max_dictionary_edit_distance for all terms lengths
    ///   Some([4].into()): max_dictionary_edit_distance for all terms lengths >= 4,
    ///   Some([2,8].into()): max_dictionary_edit_distance for all terms lengths >=2, max_dictionary_edit_distance +1 for all terms for lengths>=8
    ///   ⚠️ The resulting maximum edit distance defined in lookup() for each term length (max_edit_distance + term_length_threshold) must be
    ///   <= the corresponding maximum edit distance defined for each term length in SymSpell::new() used for creation of dictionary structures.
    /// * `prefix_length` - The length of word prefixes, from which deletes are generated. (5..7).
    /// * `count_threshold` - The minimum frequency count for dictionary words to be considered a valid for spelling correction.
    pub fn new(
        max_dictionary_edit_distance: usize,
        term_length_threshold: Option<Vec<usize>>,
        prefix_length: usize,
        count_threshold: usize,
    ) -> Self {
        Self {
            max_dictionary_edit_distance, //2
            term_length_threshold,        //vec![]
            prefix_length,                //7
            count_threshold,              //1
            corpus_word_count: 1_024_908_267_229,
            max_dictionary_term_length: 0,
            deletes: AHashMap::new(),
            words: AHashMap::new(),
            bigrams: AHashMap::new(),
            bigram_min_count: usize::MAX,
        }
    }

    /// Get the number of entries in the dictionary.
    pub fn get_dictionary_size(&self) -> usize {
        self.words.len()
    }

    /// Write the dictionary to a CSV file.
    /// Useful when the dictionary was incrementally built/updated with create_dictionary_entry.
    /// Entries are sorted by frequency count descending.
    ///
    /// # Arguments
    ///
    /// * `path` - The path+filename of the file.
    /// * `separator` - Separator between word and frequency
    pub fn save_dictionary(
        &self,
        path: &Path,
        separator: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        for (entry, count) in self
            .words
            .iter()
            .sorted_unstable_by(|a, b| Ord::cmp(&b.1, &a.1))
        {
            writeln!(writer, "{}{}{}", entry, separator, count)?;
        }
        writer.flush()?;

        Ok(())
    }

    /// Load multiple dictionary entries from a file of word/frequency count pairs.
    ///
    /// # Arguments
    ///
    /// * `corpus` - The path+filename of the file.
    /// * `term_index` - The column position of the word.
    /// * `count_index` - The column position of the frequency count.
    /// * `separator` - Separator between word and frequency
    pub fn load_dictionary(
        &mut self,
        path: &Path,
        term_index: usize,
        count_index: usize,
        separator: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let sr = BufReader::new(file);

        for line in sr.lines() {
            let line_str = line?;
            self.load_dictionary_line(&line_str, term_index, count_index, separator);
        }
        Ok(())
    }

    /// Load single dictionary entry from word/frequency count pair.
    ///
    /// # Arguments
    ///
    /// * `line` - word/frequency pair.
    /// * `term_index` - The column position of the word.
    /// * `count_index` - The column position of the frequency count.
    /// * `separator` - Separator between word and frequency
    pub fn load_dictionary_line(
        &mut self,
        line: &str,
        term_index: usize,
        count_index: usize,
        separator: &str,
    ) -> bool {
        let line_parts: Vec<&str> = line.split(separator).collect();
        if line_parts.len() >= 2 {
            // let key = unidecode(line_parts[term_index as usize]);
            let key = line_parts[term_index].to_string();
            let count = line_parts[count_index].parse::<usize>().unwrap();

            self.create_dictionary_entry(key, count);
        }
        true
    }

    /// Load multiple bigram entries from a file of bigram/frequency count pairs.
    /// Only used in lookup_compound for improved compound splitting/merging/correction quality.
    ///
    /// # Arguments
    ///
    /// * `corpus` - The path+filename of the file.
    /// * `term_index` - The column position of the word.
    /// * `count_index` - The column position of the frequency count.
    /// * `separator` - Separator between word and frequency
    pub fn load_bigram_dictionary(
        &mut self,
        path: &Path,
        term_index: usize,
        count_index: usize,
        separator: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let sr = BufReader::new(file);
        for line in sr.lines() {
            let line_str = line?;
            self.load_bigram_dictionary_line(&line_str, term_index, count_index, separator);
        }
        Ok(())
    }

    /// Load single dictionary entry from bigram/frequency count pair.
    ///
    /// # Arguments
    ///
    /// * `line` - bigram/frequency pair.
    /// * `term_index` - The column position of the word.
    /// * `count_index` - The column position of the frequency count.
    /// * `separator` - Separator between word and frequency
    pub fn load_bigram_dictionary_line(
        &mut self,
        line: &str,
        term_index: usize,
        count_index: usize,
        separator: &str,
    ) -> bool {
        let line_parts: Vec<&str> = line.split(separator).collect();
        let line_parts_len = if separator == " " { 3 } else { 2 };
        if line_parts.len() >= line_parts_len {
            let key = if separator == " " {
                [line_parts[term_index], line_parts[term_index + 1]].join(" ")
            } else {
                line_parts[term_index].to_string()
            };
            let count = line_parts[count_index].parse::<usize>().unwrap();
            self.bigrams.insert(key.into_boxed_str(), count);
            if count < self.bigram_min_count {
                self.bigram_min_count = count;
            }
        }
        true
    }

    /// Find suggested spellings for a given input word, using the maximum
    /// edit distance specified during construction of the SymSpell dictionary.
    /// Returned suggestions are sorted by distance ascending, then by frequency count descending.
    ///
    /// # Arguments
    ///
    /// * `input` - The word being spell checked. Upper/lower case allowed.
    /// * `verbosity` - The value controlling the quantity/closeness of the retuned suggestions.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    /// * `term_length_threshold` - Term length thresholds for each edit distance.
    ///   None: max_dictionary_edit_distance for all terms lengths
    ///   Some([4].into()): max_dictionary_edit_distance for all terms lengths >= 4,
    ///   Some([2,8].into()): max_dictionary_edit_distance for all terms lengths >=2, max_dictionary_edit_distance +1 for all terms for lengths>=8
    ///   ⚠️ The resulting maximum edit distance defined in lookup() for each term length (max_edit_distance + term_length_threshold) must be
    ///   <= the corresponding maximum edit distance defined for each term length in SymSpell::new() used for creation of dictionary structures.
    /// * `max_results` - Optional parameter to limit the number of suggestions returned.
    /// * `preserve_case` - Whether to preserve the letter case from input to suggestion.
    ///
    /// # Examples
    ///
    /// ```
    /// use symspell_rs::{SymSpell, Verbosity};
    /// use std::path::Path;
    ///
    /// let mut symspell: SymSpell = SymSpell::new(2,None, 7, 1);
    /// symspell.load_dictionary(Path::new("data/frequency_dictionary_en_82_765.txt"), 0, 1, " ");
    /// symspell.lookup("whatver", Verbosity::Top, 2,&None,None,false);
    /// ```
    pub fn lookup(
        &self,
        input: &str,
        verbosity: Verbosity,
        max_edit_distance: usize,
        term_length_threshold: &Option<Vec<usize>>,
        max_results: Option<usize>,
        preserve_case: bool,
    ) -> Vec<Suggestion> {
        //todo: validate term_length_threshold
        if max_edit_distance + term_length_threshold.as_ref().map_or(0, |x| x.len())
            > self.max_dictionary_edit_distance
                + self.term_length_threshold.as_ref().map_or(0, |x| x.len())
        {
            println!("max_edit_distance is bigger than max_dictionary_edit_distance");
        }

        //len(input)
        //check it for all term lengths vs check it only for current length? eigentlich reicht current, aber dann taucht der fehler später wieder auf

        //max_term_edit_distance dependent on term_length_threshold
        let max_term_edit_distance =
            term_length_threshold
                .as_ref()
                .map_or(max_edit_distance, |term_length_threshold| {
                    let term_len = len(input);
                    let mut max_term_edit_distance = 0;
                    for (i, threshold) in term_length_threshold.iter().enumerate() {
                        if term_len >= *threshold {
                            max_term_edit_distance = max_edit_distance + i
                        } else {
                            break;
                        }
                    }
                    max_term_edit_distance
                });

        let mut suggestions: Vec<Suggestion> = Vec::new();

        let input_lower_case = input.to_lowercase();
        let input_original_case = if preserve_case {
            input
        } else {
            &input_lower_case
        };
        let input = input_lower_case.as_str();

        let input_len = len(input);
        // early termination - word is too big to possibly match any words
        if input_len as isize - max_term_edit_distance as isize
            > self.max_dictionary_term_length as isize
        {
            return suggestions;
        }

        let mut hashset1: AHashSet<String> = AHashSet::new();
        let mut hashset2: AHashSet<String> = AHashSet::new();

        if self.words.contains_key(input) {
            let suggestion_count = self.words[input];
            suggestions.push(Suggestion::new(input_original_case, 0, suggestion_count));
            // early termination - return exact match, unless caller wants all matches
            if verbosity != Verbosity::All {
                return suggestions;
            }
        }

        //early termination, if we only want to check if word in dictionary or get its frequency e.g. for word segmentation
        if max_term_edit_distance == 0 {
            return suggestions;
        }

        hashset2.insert(input.to_string());

        let mut max_edit_distance2 = max_term_edit_distance;
        let mut candidate_pointer = 0;
        let mut candidates = Vec::new();

        let mut input_prefix_len = input_len;

        if input_prefix_len > self.prefix_length {
            input_prefix_len = self.prefix_length;
            candidates.push(slice(input, 0, input_prefix_len));
        } else {
            candidates.push(input.to_string());
        }

        while candidate_pointer < candidates.len() {
            let candidate = &candidates.get(candidate_pointer).unwrap().clone();
            candidate_pointer += 1;
            let candidate_len = len(candidate);
            let length_diff = input_prefix_len as isize - candidate_len as isize;

            //save some time - early termination
            //if canddate distance is already higher than suggestion distance, than there are no better suggestions to be expected
            if length_diff > max_edit_distance2 as isize {
                // skip to next candidate if Verbosity.All, look no further if Verbosity.Top or Closest
                // (candidates are ordered by delete distance, so none are closer than current)
                if verbosity == Verbosity::All {
                    continue;
                }
                break;
            }

            //read candidate entry from dictionary
            let hash = hash32(candidate.as_bytes());
            if self.deletes.contains_key(&hash) {
                let dict_suggestions = &self.deletes[&hash];

                //iterate through suggestions (to other correct dictionary items) of delete item and add them to suggestion list
                for suggestion in dict_suggestions {
                    let suggestion_len = len(suggestion);

                    if suggestion.as_ref() == input {
                        continue;
                    }

                    if suggestion_len.abs_diff(input_len) > max_edit_distance2
                        || suggestion_len < candidate_len
                        || (suggestion_len == candidate_len && suggestion.as_ref() != candidate)
                    {
                        continue;
                    }

                    let sugg_prefix_len = min(suggestion_len, self.prefix_length);

                    if sugg_prefix_len > input_prefix_len
                        && sugg_prefix_len - candidate_len > max_edit_distance2
                    {
                        continue;
                    }

                    //Damerau-Levenshtein Edit Distance: adjust distance, if both distances>0
                    //We allow simultaneous edits (deletes) of maxEditDistance on on both the dictionary and the input term.
                    //For replaces and adjacent transposes the resulting edit distance stays <= maxEditDistance.
                    //For inserts and deletes the resulting edit distance might exceed maxEditDistance.
                    //To prevent suggestions of a higher edit distance, we need to calculate the resulting edit distance, if there are simultaneous edits on both sides.
                    //Example: (bank==bnak and bank==bink, but bank!=kanb and bank!=xban and bank!=baxn for maxEditDistance=1)
                    //Two deletes on each side of a pair makes them all equal, but the first two pairs have edit distance=1, the others edit distance=2.
                    let distance;
                    if candidate_len == 0 {
                        //suggestions which have no common chars with input (inputLen<=maxEditDistance && suggestionLen<=maxEditDistance)
                        distance = cmp::max(input_len, suggestion_len);

                        if distance > max_edit_distance2 || hashset2.contains(suggestion.as_ref()) {
                            continue;
                        }
                        hashset2.insert(suggestion.to_string());
                    } else if suggestion_len == 1 {
                        distance = if !input.contains(&slice(suggestion, 0, 1)) {
                            input_len
                        } else {
                            input_len - 1
                        };

                        if distance > max_edit_distance2 || hashset2.contains(suggestion.as_ref()) {
                            continue;
                        }

                        hashset2.insert(suggestion.to_string());
                    // number of edits in prefix ==maxediddistance  AND no identic suffix,
                    // then editdistance>maxEditDistance and no need for Levenshtein calculation
                    // (inputLen >= prefixLength) && (suggestionLen >= prefixLength)
                    } else if self.has_different_suffix(
                        max_term_edit_distance,
                        input,
                        input_len,
                        candidate_len,
                        suggestion,
                        suggestion_len,
                    ) {
                        continue;
                    } else {
                        // DeleteInSuggestionPrefix is somewhat expensive, and only pays off when verbosity is Top or Closest.
                        if verbosity != Verbosity::All
                            && !self.delete_in_suggestion_prefix(
                                candidate,
                                candidate_len,
                                suggestion,
                                suggestion_len,
                            )
                        {
                            continue;
                        }

                        if hashset2.contains(suggestion.as_ref()) {
                            continue;
                        }
                        hashset2.insert(suggestion.to_string());

                        distance = if let Some(distance) =
                            damerau_levenshtein_osa(input, suggestion, max_edit_distance2)
                        {
                            distance
                        } else {
                            continue;
                        };
                    }
                    //save some time
                    //do not process higher distances than those already found, if verbosity<All (note: maxEditDistance2 will always equal maxEditDistance when Verbosity::All)
                    if distance <= max_edit_distance2 {
                        let suggestion_count = self.words[suggestion];
                        let si = Suggestion::new(suggestion.as_ref(), distance, suggestion_count);

                        if !suggestions.is_empty() {
                            match verbosity {
                                Verbosity::Closest => {
                                    //we will calculate DamLev distance only to the smallest found distance so far
                                    if distance < max_edit_distance2 {
                                        suggestions.clear();
                                    }
                                }
                                Verbosity::Top => {
                                    if distance < max_edit_distance2
                                        || suggestion_count > suggestions[0].count
                                    {
                                        max_edit_distance2 = distance;
                                        suggestions[0] = si;
                                    }
                                    continue;
                                }
                                _ => (),
                            }
                        }

                        if verbosity != Verbosity::All {
                            max_edit_distance2 = distance;
                        }

                        suggestions.push(si);
                    }
                }
            }

            //add edits
            //derive edits (deletes) from candidate (input) and add them to candidates list
            //this is a recursive process until the maximum edit distance has been reached
            if length_diff < max_term_edit_distance as isize && candidate_len <= self.prefix_length
            {
                //save some time
                //do not create edits with edit distance smaller than suggestions already found
                if verbosity != Verbosity::All && length_diff >= max_edit_distance2 as isize {
                    continue;
                }

                for i in 0..candidate_len {
                    let delete = remove(candidate, i);

                    if !hashset1.contains(&delete) {
                        hashset1.insert(delete.clone());
                        candidates.push(delete);
                    }
                }
            }
        }

        //sort by ascending edit distance, then by descending word frequency
        if suggestions.len() > 1 {
            suggestions.sort_unstable();
        }

        //transfer case from input to suggestion
        if preserve_case {
            for suggestion in suggestions.iter_mut() {
                suggestion.term = transfer_case(input_original_case, &suggestion.term);
            }
        }

        if let Some(max_results) = max_results {
            suggestions.truncate(max_results);
        }
        suggestions
    }

    /// Find suggested spellings for a multi-word input string (supports word splitting/merging).
    /// Returns a list of Suggestion representing suggested correct spellings for the input string.
    ///
    /// lookup_compound supports compound aware automatic spelling correction of multi-word input strings with three cases:
    /// 1. mistakenly inserted space into a correct word led to two incorrect terms
    /// 2. mistakenly omitted space between two correct words led to one incorrect combined term
    /// 3. multiple independent input terms with/without spelling errors
    ///
    /// # Arguments
    ///
    /// * `input` - The sentence being spell checked.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    /// * `preserve_case` - Whether to preserve the letter case from input to suggestion.
    ///
    /// # Examples
    ///
    /// ```
    /// use symspell_rs::{SymSpell};
    /// use std::path::Path;
    ///
    /// let mut symspell: SymSpell = SymSpell::new(2, None,7, 1);
    /// symspell.load_dictionary(Path::new("data/frequency_dictionary_en_82_765.txt"), 0, 1, " ");
    /// symspell.lookup_compound("whereis th elove", 2, &None,false);
    /// ```
    pub fn lookup_compound(
        &self,
        input: &str,
        edit_distance_max: usize,
        term_length_threshold: &Option<Vec<usize>>,
        preserve_case: bool,
    ) -> Vec<Suggestion> {
        //parse input string into single terms
        let term_list1 = parse_words(input);

        let mut suggestions: Vec<Suggestion>; //suggestions for a single term
        let mut suggestion_parts: Vec<Suggestion> = Vec::new(); //1 line with separate parts

        //translate every term to its best suggestion, otherwise it remains unchanged
        let mut last_combi = false;
        for (i, term) in term_list1.iter().enumerate() {
            suggestions = self.lookup(
                term,
                Verbosity::Top,
                edit_distance_max,
                term_length_threshold,
                None,
                false,
            );

            //combi check, always before split
            if i > 0 && !last_combi {
                let mut suggestions_combi: Vec<Suggestion> = self.lookup(
                    &[term_list1[i - 1].as_str(), term_list1[i].as_str()].join(""),
                    Verbosity::Top,
                    edit_distance_max,
                    term_length_threshold,
                    None,
                    false,
                );

                if !suggestions_combi.is_empty() {
                    let best1 = suggestion_parts[suggestion_parts.len() - 1].clone();
                    let best2 = if !suggestions.is_empty() {
                        suggestions[0].clone()
                    } else {
                        Suggestion::new(
                            //unknown word
                            term_list1[i].as_str(),
                            //estimated edit distance
                            edit_distance_max + 1,
                            // estimated word occurrence probability P=10 / (N * 10^word length l)
                            // estimated word count C=10 / 10^word length l
                            // formulae to calculate the probability of an unknown word proposed by Peter Norvig in Natural Language Corpus Data, page 224 http://norvig.com/ngrams/ch14.pdf
                            // estimated count always 0 if termlength > 3 and independent from corpus_word_count???
                            (10f64 / 10usize.saturating_pow(len(&term_list1[i]) as u32) as f64)
                                as usize,
                        )
                    };

                    //distance1=edit distance between 2 split terms und their best corrections : as comparative value for the combination
                    let distance1 = best1.distance + best2.distance;
                    if suggestions_combi[0].distance + 1 < distance1
                        || (suggestions_combi[0].distance + 1 == distance1
                            && (suggestions_combi[0].count
                                    // best1 / corpus * best1 / corpus * corpus
                                    > (best1.count as f64 / self.corpus_word_count as f64 * best2.count as f64) as usize))
                    {
                        suggestions_combi[0].distance += 1;
                        let last_i = suggestion_parts.len() - 1;
                        suggestion_parts[last_i] = suggestions_combi[0].clone();
                        last_combi = true;
                        continue;
                    }
                }
            }
            last_combi = false;

            //alway split terms without suggestion / never split terms with suggestion ed=0 / never split single char terms
            if !suggestions.is_empty()
                && ((suggestions[0].distance == 0) || (len(&term_list1[i]) == 1))
            {
                //choose best suggestion
                suggestion_parts.push(suggestions[0].clone());
            } else {
                let mut suggestion_split_best = if !suggestions.is_empty() {
                    //add original term
                    suggestions[0].clone()
                } else {
                    //if no perfect suggestion, split word into pairs
                    Suggestion::empty()
                };

                let term_length = len(&term_list1[i]);
                if term_length > 1 {
                    for j in 1..term_length {
                        let part1 = slice(&term_list1[i], 0, j);
                        let part2 = slice(&term_list1[i], j, term_length);
                        let mut suggestion_split = Suggestion::empty();
                        let suggestions1 = self.lookup(
                            &part1,
                            Verbosity::Top,
                            edit_distance_max,
                            term_length_threshold,
                            None,
                            false,
                        );
                        if !suggestions1.is_empty() {
                            let suggestions2 = self.lookup(
                                &part2,
                                Verbosity::Top,
                                edit_distance_max,
                                term_length_threshold,
                                None,
                                false,
                            );

                            if !suggestions2.is_empty() {
                                //select best suggestion for split pair
                                suggestion_split.term =
                                    [suggestions1[0].term.as_str(), suggestions2[0].term.as_str()]
                                        .join(" ");

                                /*
                                let mut distance2 = damerau_levenshtein_osa(
                                    &term_list1[i],
                                    &suggestion_split.term,
                                    edit_distance_max as usize,
                                );

                                if distance2 < 0 {
                                    distance2 = edit_distance_max + 1;
                                }
                                */

                                let distance2 = if let Some(d) = damerau_levenshtein_osa(
                                    &term_list1[i],
                                    &suggestion_split.term,
                                    edit_distance_max,
                                ) {
                                    d
                                } else {
                                    edit_distance_max + 1
                                };

                                if !suggestion_split_best.term.is_empty() {
                                    if distance2 > suggestion_split_best.distance {
                                        continue;
                                    }
                                    if distance2 < suggestion_split_best.distance {
                                        suggestion_split_best = Suggestion::empty();
                                    }
                                }

                                let bigram_count: usize =
                                    match self.bigrams.get(&*suggestion_split.term) {
                                        //if bigram exists in bigram dictionary
                                        Some(&bigram_frequency) => {
                                            //increase count, if split.corrections are part of or identical to input
                                            //single term correction exists
                                            if !suggestions.is_empty() {
                                                let best_si = &suggestions[0];
                                                //alternatively remove the single term from suggestionsSplit, but then other splittings could win
                                                if suggestion_split.term == term_list1[i] {
                                                    //make count bigger than count of single term correction
                                                    cmp::max(bigram_frequency, best_si.count + 2)
                                                } else if suggestions1[0].term == best_si.term
                                                    || suggestions2[0].term == best_si.term
                                                {
                                                    //make count bigger than count of single term correction
                                                    cmp::max(bigram_frequency, best_si.count + 1)
                                                } else {
                                                    bigram_frequency
                                                }
                                            // no single term correction exists
                                            } else if suggestion_split.term == term_list1[i] {
                                                cmp::max(
                                                    bigram_frequency,
                                                    cmp::max(
                                                        suggestions1[0].count,
                                                        suggestions2[0].count,
                                                    ) + 2,
                                                )
                                            } else {
                                                bigram_frequency
                                            }
                                        }
                                        None => {
                                            //The Naive Bayes probability of the word combination is the product of the two word probabilities: P(AB) = P(A) * P(B)
                                            //use it to estimate the frequency count of the combination if no bigram in dictionary found, which then is used to rank/select the best splitting variant
                                            min(
                                                self.bigram_min_count,
                                                (suggestions1[0].count as f64
                                                    / self.corpus_word_count as f64
                                                    * suggestions2[0].count as f64)
                                                    as usize,
                                            )
                                        }
                                    };

                                suggestion_split.distance = distance2;
                                suggestion_split.count = bigram_count;

                                if suggestion_split_best.term.is_empty()
                                    || suggestion_split.count > suggestion_split_best.count
                                {
                                    suggestion_split_best = suggestion_split.clone();
                                }
                            }
                        }
                    }

                    if !suggestion_split_best.term.is_empty() {
                        //select best suggestion for split pair
                        suggestion_parts.push(suggestion_split_best.clone());
                    } else {
                        let mut si = Suggestion::empty();
                        si.term = term_list1[i].clone();
                        // estimated word occurrence probability P=10 / (N * 10^word length l)
                        // estimated word count C=10 / 10^word length l
                        // formulae to calculate the probability of an unknown word proposed by Peter Norvig in Natural Language Corpus Data, page 224 http://norvig.com/ngrams/ch14.pdf
                        // estimated count always 0 if termlength > 3 and independent from corpus_word_count???
                        si.count =
                            (10f64 / 10usize.saturating_pow(term_length as u32) as f64) as usize;
                        si.distance = edit_distance_max + 1;
                        suggestion_parts.push(si);
                    }
                } else {
                    let mut si = Suggestion::empty();
                    si.term = term_list1[i].clone();
                    // estimated word occurrence probability P=10 / (N * 10^word length l)
                    // estimated word count C=10 / 10^word length l
                    // formulae to calculate the probability of an unknown word proposed by Peter Norvig in Natural Language Corpus Data, page 224 http://norvig.com/ngrams/ch14.pdf
                    // estimated count always 0 if termlength > 3 and independent from corpus_word_count???
                    si.count = (10f64 / 10usize.saturating_pow(term_length as u32) as f64) as usize;
                    si.distance = edit_distance_max + 1;
                    suggestion_parts.push(si);
                }
            }
        }

        let mut suggestion = Suggestion::empty();

        let mut tmp_count: f64 = self.corpus_word_count as f64;

        let mut s = "".to_string();
        for si in suggestion_parts {
            s.push_str(&si.term);
            s.push(' ');
            tmp_count *= si.count as f64 / self.corpus_word_count as f64;
        }

        let output = s.trim();
        suggestion.count = tmp_count as usize;
        suggestion.distance =
            damerau_levenshtein_osa(&input.to_lowercase(), output, usize::MAX).unwrap();

        //transfer case from input to suggestion
        suggestion.term = if preserve_case {
            transfer_case(input, output)
        } else {
            output.to_lowercase()
        };

        vec![suggestion]
    }

    /// word_segmentation divides a string into words by inserting missing spaces at the appropriate positions.
    /// word_segmentation works on text with any letter case which is retained in the output segmentation.
    /// word_segmentation works on noisy text with spelling mistakes, which are corrected in the output segmentation.
    /// existing spaces are allowed and considered for optimum segmentation.
    ///
    /// word_segmentation uses a novel approach *without* recursion.
    /// https://seekstorm.com/blog/fast-word-segmentation-noisy-text/
    /// While each string of length n can be segmentend in 2^n−1 possible compositions https://en.wikipedia.org/wiki/Composition_(combinatorics)
    /// word_segmentation has a linear runtime O(n) to find the optimum composition
    ///
    /// # Arguments
    ///
    /// * `input` - The string being segmented into words. Upper/lower case allowed.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Returns
    ///
    /// * the word segmented and spelling corrected string,
    /// * The edit distance sum between input string and corrected string,
    /// * The sum of word occurence probabilities in log scale (a measure of how common and probable the corrected segmentation is).
    ///
    /// # Examples
    ///
    /// ```
    /// use symspell_rs::{SymSpell, Verbosity};
    /// use std::path::Path;
    ///
    /// let mut symspell: SymSpell = SymSpell::new(2, None, 7, 1);
    /// symspell.load_dictionary(Path::new("data/frequency_dictionary_en_82_765.txt"), 0, 1, " ");
    /// symspell.word_segmentation("itwas", 2);
    /// ```
    pub fn word_segmentation(&self, input: &str, max_edit_distance: usize) -> Composition {
        // Normalize ligatures: "scientiﬁc" "ﬁelds" "ﬁnal"
        let input = &unicode_normalization_form_kc(input).replace('\u{002D}', ""); // Remove U+002D (hyphen-minus);

        let asize = len(input);

        let mut ci: usize = 0;
        let mut compositions: Vec<Composition> = vec![Composition::empty(); asize];

        //outer loop (column): all possible part start positions
        for j in 0..asize {
            //inner loop (row): all possible part lengths (from start position): part can't be bigger than longest word in dictionary (other than long unknown word)
            let imax = min(asize - j, self.max_dictionary_term_length);
            for i in 1..=imax {
                //get top spelling correction/ed for part
                let mut part = slice(input, j, j + i);

                let mut sep_len = 0;
                let mut top_ed = 0;

                let first_char = at(&part, 0).unwrap();
                if first_char.is_whitespace() {
                    //remove space for levensthein calculation
                    part = remove(&part, 0);
                } else {
                    //add ed+1: space did not exist, had to be inserted
                    sep_len = 1;
                }

                //remove space from part1, add number of removed spaces to topEd
                top_ed += part.len();
                //remove space
                part = part.replace(" ", "");
                top_ed -= part.len();

                // Lookup against the lowercase term
                // word_segmentation works on text with any case which is retained in the output segmentation.
                // word_segmentation works on noisy text with spelling mistakes, which are corrected in the output segmentation.
                let results =
                    self.lookup(&part, Verbosity::Top, max_edit_distance, &None, None, true);
                let top_prob_log = if !results.is_empty() {
                    //retain/preserve/transfer letter case during correction
                    if results[0].distance > 0 {
                        part = results[0].term.clone();
                        top_ed += results[0].distance;
                    }

                    //Naive Bayes Rule
                    //we assume the word probabilities of two words to be independent
                    //therefore the resulting probability of the word combination is the product of the two word probabilities

                    //instead of computing the product of probabilities we are computing the sum of the logarithm of probabilities
                    //because the probabilities of words are about 10^-10, the product of many such small numbers could exceed (underflow) the floating number range and become zero
                    //log(ab)=log(a)+log(b)
                    //todo: use decimal crate for higher precision?
                    (results[0].count as f64 / self.corpus_word_count as f64).log10()
                } else {
                    let part_len = len(&part);

                    //default, if word not found
                    //otherwise long input text would win as long unknown word (with ed=edmax+1 ), although there there should many spaces inserted
                    top_ed += part_len;
                    (10.0 / (self.corpus_word_count as f64 * 10.0f64.powf(part_len as f64))).log10()
                };

                let di = (i + ci) % asize;
                // set values in first loop
                if j == 0 {
                    compositions[i - 1] = Composition {
                        segmented_string: part.to_owned(),
                        distance_sum: top_ed,
                        prob_log_sum: top_prob_log,
                    };
                } else if i == self.max_dictionary_term_length
                    //replace values if better probabilityLogSum, if same edit distance OR one space difference 
                    || (((compositions[ci].distance_sum + top_ed == compositions[di].distance_sum)
                        || (compositions[ci].distance_sum + sep_len + top_ed
                            == compositions[di].distance_sum))
                        && (compositions[di].prob_log_sum
                            < compositions[ci].prob_log_sum + top_prob_log))
                    //replace values if smaller edit distance 
                    || (compositions[ci].distance_sum + sep_len + top_ed
                        < compositions[di].distance_sum)
                {
                    //keep punctuation or apostrophe adjacent to previous word
                    if ((part.len() == 1) && part.chars().nth(0).unwrap().is_ascii_punctuation())
                        || ((part.len() == 3) && part.starts_with("’"))
                    {
                        compositions[di] = Composition {
                            segmented_string: [
                                compositions[ci].segmented_string.as_str(),
                                part.as_str(),
                            ]
                            .join(""),
                            distance_sum: compositions[ci].distance_sum + top_ed,
                            prob_log_sum: compositions[ci].prob_log_sum + top_prob_log,
                        };
                    } else {
                        //todo: keep segmented_string and corrected string separate
                        compositions[di] = Composition {
                            segmented_string: [
                                compositions[ci].segmented_string.as_str(),
                                part.as_str(),
                            ]
                            .join(" "),
                            distance_sum: compositions[ci].distance_sum + sep_len + top_ed,
                            prob_log_sum: compositions[ci].prob_log_sum + top_prob_log,
                        };
                    }
                }
            }
            if j != 0 {
                ci += 1;
            }
            ci = if ci == asize { 0 } else { ci };
        }
        compositions[ci].to_owned()
    }

    // Check whether all delete chars are present in the suggestion prefix in correct order, otherwise this is just a hash collision
    fn delete_in_suggestion_prefix(
        &self,
        delete: &str,
        delete_len: usize,
        suggestion: &str,
        suggestion_len: usize,
    ) -> bool {
        if delete_len == 0 {
            return true;
        }
        let suggestion_len = if self.prefix_length < suggestion_len {
            self.prefix_length
        } else {
            suggestion_len
        };
        let mut j = 0;
        for i in 0..delete_len {
            let del_char = at(delete, i as isize).unwrap();
            while j < suggestion_len && del_char != at(suggestion, j as isize).unwrap() {
                j += 1;
            }

            if j == suggestion_len {
                return false;
            }
        }
        true
    }

    /// Create/Update an entry in the dictionary
    /// For every word there are deletes with an edit distance of 1..maxEditDistance created and added to the
    /// dictionary. Every delete entry has a suggestions list, which points to the original term(s) it was created from.
    /// The dictionary may be dynamically updated (word frequency and new words) at any time by calling CreateDictionaryEntry
    /// # Arguments
    ///
    /// * `term` - The word to add to dictionary.
    /// * `count` - The frequency count for word.
    ///
    /// Returns True if the word is added or updated and the sum count reaches or exceeds the threshold for the first time,
    /// making it a new entry valid for suggestions, otherwise False.
    pub fn create_dictionary_entry<T>(&mut self, term: T, count: usize) -> bool
    where
        T: Clone + AsRef<str> + Into<String>,
    {
        // update words
        let entry = self
            .words
            .entry(term.clone().into().into_boxed_str())
            .or_insert(0);
        if *entry == 0 {
            *entry = count;
            if count < self.count_threshold {
                return false;
            }
        } else {
            let old_count = *entry;
            let updated_count = if usize::MAX - *entry > count {
                *entry + count
            } else {
                usize::MAX
            };
            *entry = updated_count;

            if updated_count < self.count_threshold || old_count >= self.count_threshold {
                return false;
            }
        }

        // create deletes
        let term_len = len(term.as_ref());

        //do not create deletes for short words below threshold
        //todo: move to start of function: latency of len vs. words entry + count?
        if self
            .term_length_threshold
            .as_ref()
            .is_some_and(|term_length_threshold| {
                !term_length_threshold.is_empty() && term_len < term_length_threshold[0]
            })
        {
            return false;
        }

        let max_term_edit_distance = self.term_length_threshold.as_ref().map_or(
            self.max_dictionary_edit_distance,
            |term_length_threshold| {
                //max_term_edit_distance dependent on term_length_threshold
                let mut max_term_edit_distance = 0;
                for (i, threshold) in term_length_threshold.iter().enumerate() {
                    if term_len >= *threshold {
                        max_term_edit_distance = self.max_dictionary_edit_distance + i
                    } else {
                        break;
                    }
                }
                max_term_edit_distance
            },
        );

        //collect max_dictionary_term_length
        if term_len > self.max_dictionary_term_length {
            self.max_dictionary_term_length = term_len;
        }

        let edits = self.edits_prefix(term.as_ref(), max_term_edit_distance);

        for delete in edits {
            let delete_hash = hash32(delete.as_bytes());

            self.deletes
                .entry(delete_hash)
                .and_modify(|e| e.push(term.clone().into().into_boxed_str()))
                .or_insert_with(|| vec![term.clone().into().into_boxed_str()]);
        }

        true
    }

    fn edits_prefix(&self, key: &str, max_term_edit_distance: usize) -> AHashSet<String> {
        let mut hash_set = AHashSet::new();

        let key_len = len(key);

        if key_len <= self.max_dictionary_edit_distance {
            hash_set.insert("".to_string());
        }

        if key_len > self.prefix_length {
            let shortened_key = slice(key, 0, self.prefix_length);
            hash_set.insert(shortened_key.clone());
            edits(&shortened_key, 0, max_term_edit_distance, &mut hash_set);
        } else {
            hash_set.insert(key.to_string());
            edits(key, 0, max_term_edit_distance, &mut hash_set);
        };

        hash_set
    }

    fn has_different_suffix(
        &self,
        max_edit_distance: usize,
        input: &str,
        input_len: usize,
        candidate_len: usize,
        suggestion: &str,
        suggestion_len: usize,
    ) -> bool {
        // handles the shortcircuit of min_distance
        // assignment when first boolean expression
        // evaluates to false
        let min =
            if self.prefix_length as isize - max_edit_distance as isize == candidate_len as isize {
                min(input_len, suggestion_len) as isize - self.prefix_length as isize
            } else {
                0
            };

        (self.prefix_length - max_edit_distance == candidate_len)
            && (((min - self.prefix_length as isize) > 1)
                && (suffix(input, input_len + 1 - min as usize)
                    != suffix(suggestion, suggestion_len + 1 - min as usize)))
            || ((min > 0)
                && (at(input, (input_len - min as usize) as isize)
                    != at(suggestion, (suggestion_len - min as usize) as isize))
                && ((at(input, (input_len - min as usize - 1) as isize)
                    != at(suggestion, (suggestion_len - min as usize) as isize))
                    || (at(input, (input_len - min as usize) as isize)
                        != at(suggestion, (suggestion_len - min as usize - 1) as isize))))
    }
}

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

#[cfg(not(all(target_feature = "aes", target_feature = "sse2")))]
use ahash::RandomState;
use ahash::{AHashMap, AHashSet};
use std::cmp;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
#[cfg(not(all(target_feature = "aes", target_feature = "sse2")))]
use std::sync::LazyLock;

#[cfg(not(all(target_feature = "aes", target_feature = "sse2")))]
pub static HASHER_64: LazyLock<RandomState> =
    LazyLock::new(|| RandomState::with_seeds(808259318, 750368348, 84901999, 789810389));

// stable hash, faster, but not available on all platforms
// https://github.com/tkaitchuck/aHash
#[inline]
#[cfg(all(target_feature = "aes", target_feature = "sse2"))]
pub(crate) fn hash64(term_bytes: &[u8]) -> u64 {
    use gxhash::gxhash64;

    gxhash64(term_bytes, 1234)
}

// unstable hash, slower, but available on all platforms
// https://github.com/ogxd/gxhash
#[inline]
#[cfg(not(all(target_feature = "aes", target_feature = "sse2")))]
pub(crate) fn hash64(term_bytes: &[u8]) -> u64 {
    HASHER_64.hash_one(term_bytes)
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
pub fn damerau_levenshtein_osa(a: &str, b: &str, max_distance: usize) -> i64 {
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
        prev_distances[b_len] as i64
    } else {
        -1
    }
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
    pub segmented_string: String,
    pub distance_sum: i64,
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
pub struct Suggestion {
    pub term: String,
    pub distance: i64,
    pub count: i64,
}

impl Suggestion {
    pub fn empty() -> Suggestion {
        Suggestion {
            term: "".to_string(),
            distance: 0,
            count: 0,
        }
    }

    pub fn new(term: impl Into<String>, distance: i64, count: i64) -> Suggestion {
        Suggestion {
            term: term.into(),
            distance,
            count,
        }
    }
}

impl Ord for Suggestion {
    fn cmp(&self, other: &Suggestion) -> Ordering {
        let distance_cmp = self.distance.cmp(&other.distance);
        if distance_cmp == Ordering::Equal {
            return self.count.cmp(&other.count);
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
    /// Maximum edit distance for doing lookups.
    max_dictionary_edit_distance: i64,
    /// The length of word prefixes used for spell checking.
    prefix_length: i64,
    /// The minimum frequency count for dictionary words to be considered correct spellings.
    count_threshold: i64,

    //// number of all words in the corpus used to generate the
    //// frequency dictionary. This is used to calculate the word
    //// occurrence probability p from word counts c : p=c/N. N equals
    //// the sum of all counts c in the dictionary only if the
    //// dictionary is complete, but not if the dictionary is
    //// truncated or filtered
    corpus_word_count: i64,
    max_length: i64,
    deletes: AHashMap<u64, Vec<Box<str>>>,
    words: AHashMap<Box<str>, i64>,
    bigrams: AHashMap<Box<str>, i64>,
    bigram_min_count: i64,
}

impl SymSpell {
    /// Creates a new SymSpell instance.
    pub fn new(
        max_dictionary_edit_distance: i64,
        prefix_length: i64,
        count_threshold: i64,
    ) -> Self {
        Self {
            max_dictionary_edit_distance, //2
            prefix_length,                //7
            count_threshold,              //1
            corpus_word_count: 1_024_908_267_229,
            max_length: 0,
            deletes: AHashMap::new(),
            words: AHashMap::new(),
            bigrams: AHashMap::new(),
            bigram_min_count: i64::MAX,
        }
    }

    /// Get the number of entries in the dictionary.
    pub fn get_dictionary_size(&self) -> usize {
        self.words.len()
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
        corpus: &str,
        term_index: i64,
        count_index: i64,
        separator: &str,
    ) -> bool {
        if !Path::new(corpus).exists() {
            return false;
        }

        let file = File::open(corpus).expect("file not found");
        let sr = BufReader::new(file);

        for line in sr.lines() {
            let line_str = line.unwrap();
            self.load_dictionary_line(&line_str, term_index, count_index, separator);
        }
        true
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
        term_index: i64,
        count_index: i64,
        separator: &str,
    ) -> bool {
        let line_parts: Vec<&str> = line.split(separator).collect();
        if line_parts.len() >= 2 {
            // let key = unidecode(line_parts[term_index as usize]);
            let key = line_parts[term_index as usize].to_string();
            let count = line_parts[count_index as usize].parse::<i64>().unwrap();

            self.create_dictionary_entry(key, count);
        }
        true
    }

    /// Load multiple bigram entries from a file of bigram/frequency count pairs.
    ///
    /// # Arguments
    ///
    /// * `corpus` - The path+filename of the file.
    /// * `term_index` - The column position of the word.
    /// * `count_index` - The column position of the frequency count.
    /// * `separator` - Separator between word and frequency
    pub fn load_bigram_dictionary(
        &mut self,
        corpus: &str,
        term_index: i64,
        count_index: i64,
        separator: &str,
    ) -> bool {
        if !Path::new(corpus).exists() {
            return false;
        }
        let file = File::open(corpus).expect("file not found");
        let sr = BufReader::new(file);
        for line in sr.lines() {
            let line_str = line.unwrap();
            self.load_bigram_dictionary_line(&line_str, term_index, count_index, separator);
        }
        true
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
        term_index: i64,
        count_index: i64,
        separator: &str,
    ) -> bool {
        let line_parts: Vec<&str> = line.split(separator).collect();
        let line_parts_len = if separator == " " { 3 } else { 2 };
        if line_parts.len() >= line_parts_len {
            let key = if separator == " " {
                format!(
                    "{} {}",
                    line_parts[term_index as usize],
                    line_parts[(term_index + 1) as usize]
                )
                .to_string()
            } else {
                line_parts[term_index as usize].to_string()
            };
            let count = line_parts[count_index as usize].parse::<i64>().unwrap();
            self.bigrams.insert(key.into_boxed_str(), count);
            if count < self.bigram_min_count {
                self.bigram_min_count = count;
            }
        }
        true
    }

    /// Find suggested spellings for a given input word, using the maximum
    /// edit distance specified during construction of the SymSpell dictionary.
    ///
    /// # Arguments
    ///
    /// * `input` - The word being spell checked.
    /// * `verbosity` - The value controlling the quantity/closeness of the retuned suggestions.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Examples
    ///
    /// ```
    /// use symspell_rs::{SymSpell, Verbosity};
    ///
    /// let mut symspell: SymSpell = SymSpell::new(2, 7, 1);
    /// symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    /// symspell.lookup("whatver", Verbosity::Top, 2);
    /// ```
    pub fn lookup(
        &self,
        input: &str,
        verbosity: Verbosity,
        max_edit_distance: i64,
    ) -> Vec<Suggestion> {
        if max_edit_distance > self.max_dictionary_edit_distance {
            panic!("max_edit_distance is bigger than max_dictionary_edit_distance");
        }

        let mut suggestions: Vec<Suggestion> = Vec::new();

        let prep_input = (input).to_string();
        let input = prep_input.as_str();
        let input_len = len(input) as i64;

        if input_len - self.max_dictionary_edit_distance > self.max_length {
            return suggestions;
        }

        let mut hashset1: AHashSet<String> = AHashSet::new();
        let mut hashset2: AHashSet<String> = AHashSet::new();

        if self.words.contains_key(input) {
            let suggestion_count = self.words[input];
            suggestions.push(Suggestion::new(input, 0, suggestion_count));

            if verbosity != Verbosity::All {
                return suggestions;
            }
        }

        hashset2.insert(input.to_string());

        let mut max_edit_distance2 = max_edit_distance;
        let mut candidate_pointer = 0;
        let mut candidates = Vec::new();

        let mut input_prefix_len = input_len;

        if input_prefix_len > self.prefix_length {
            input_prefix_len = self.prefix_length;
            candidates.push(slice(input, 0, input_prefix_len as usize));
        } else {
            candidates.push(input.to_string());
        }

        while candidate_pointer < candidates.len() {
            let candidate = &candidates.get(candidate_pointer).unwrap().clone();
            candidate_pointer += 1;
            let candidate_len = len(candidate) as i64;
            let length_diff = input_prefix_len - candidate_len;

            if length_diff > max_edit_distance2 {
                if verbosity == Verbosity::All {
                    continue;
                }
                break;
            }

            let hash = hash64(candidate.as_bytes());
            if self.deletes.contains_key(&hash) {
                let dict_suggestions = &self.deletes[&hash];

                for suggestion in dict_suggestions {
                    let suggestion_len = len(suggestion) as i64;

                    if suggestion.as_ref() == input {
                        continue;
                    }

                    if (suggestion_len - input_len).abs() > max_edit_distance2
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

                    let distance;

                    if candidate_len == 0 {
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
                    } else if self.has_different_suffix(
                        max_edit_distance,
                        input,
                        input_len,
                        candidate_len,
                        suggestion,
                        suggestion_len,
                    ) {
                        continue;
                    } else {
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

                        distance =
                            damerau_levenshtein_osa(input, suggestion, max_edit_distance2 as usize);

                        if distance < 0 {
                            continue;
                        }
                    }

                    if distance <= max_edit_distance2 {
                        let suggestion_count = self.words[suggestion];
                        let si = Suggestion::new(suggestion.as_ref(), distance, suggestion_count);

                        if !suggestions.is_empty() {
                            match verbosity {
                                Verbosity::Closest => {
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

            if length_diff < max_edit_distance && candidate_len <= self.prefix_length {
                if verbosity != Verbosity::All && length_diff >= max_edit_distance2 {
                    continue;
                }

                for i in 0..candidate_len {
                    let delete = remove(candidate, i as usize);

                    if !hashset1.contains(&delete) {
                        hashset1.insert(delete.clone());
                        candidates.push(delete);
                    }
                }
            }
        }

        if suggestions.len() > 1 {
            suggestions.sort();
        }

        suggestions
    }

    /// Find suggested spellings for a given input sentence, using the maximum
    /// edit distance specified during construction of the SymSpell dictionary.
    ///
    /// # Arguments
    ///
    /// * `input` - The sentence being spell checked.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Examples
    ///
    /// ```
    /// use symspell_rs::{SymSpell};
    ///
    /// let mut symspell: SymSpell = SymSpell::new(2, 7, 1);
    /// symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    /// symspell.lookup_compound("whereis th elove", 2);
    /// ```
    pub fn lookup_compound(&self, input: &str, edit_distance_max: i64) -> Vec<Suggestion> {
        //parse input string into single terms
        let term_list1 = self.parse_words(input);

        // let mut suggestions_previous_term: Vec<Suggestion> = Vec::new();                  //suggestions for a single term
        let mut suggestions: Vec<Suggestion>;
        let mut suggestion_parts: Vec<Suggestion> = Vec::new();

        //translate every term to its best suggestion, otherwise it remains unchanged
        let mut last_combi = false;

        for (i, term) in term_list1.iter().enumerate() {
            suggestions = self.lookup(term, Verbosity::Top, edit_distance_max);

            //combi check, always before split
            if i > 0 && !last_combi {
                let mut suggestions_combi: Vec<Suggestion> = self.lookup(
                    &format!("{}{}", term_list1[i - 1], term_list1[i]),
                    Verbosity::Top,
                    edit_distance_max,
                );

                if !suggestions_combi.is_empty() {
                    let best1 = suggestion_parts[suggestion_parts.len() - 1].clone();
                    let best2 = if !suggestions.is_empty() {
                        suggestions[0].clone()
                    } else {
                        Suggestion::new(
                            term_list1[1].as_str(),
                            edit_distance_max + 1,
                            10 / (10i64).pow(len(&term_list1[i]) as u32),
                        )
                    };

                    //if (suggestions_combi[0].distance + 1 < DamerauLevenshteinDistance(term_list1[i - 1] + " " + term_list1[i], best1.term + " " + best2.term))
                    let distance1 = best1.distance + best2.distance;

                    if (distance1 >= 0)
                        && (suggestions_combi[0].distance + 1 < distance1
                            || (suggestions_combi[0].distance + 1 == distance1
                                && (suggestions_combi[0].count
                                    > best1.count / self.corpus_word_count * best2.count)))
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

                        let suggestions1 = self.lookup(&part1, Verbosity::Top, edit_distance_max);

                        if !suggestions1.is_empty() {
                            let suggestions2 =
                                self.lookup(&part2, Verbosity::Top, edit_distance_max);

                            if !suggestions2.is_empty() {
                                //select best suggestion for split pair
                                suggestion_split.term =
                                    format!("{} {}", suggestions1[0].term, suggestions2[0].term);

                                let mut distance2 = damerau_levenshtein_osa(
                                    &term_list1[i],
                                    &format!("{} {}", suggestions1[0].term, suggestions2[0].term),
                                    edit_distance_max as usize,
                                );

                                if distance2 < 0 {
                                    distance2 = edit_distance_max + 1;
                                }

                                if !suggestion_split_best.term.is_empty() {
                                    if distance2 > suggestion_split_best.distance {
                                        continue;
                                    }
                                    if distance2 < suggestion_split_best.distance {
                                        suggestion_split_best = Suggestion::empty();
                                    }
                                }
                                let count2: i64 = match self.bigrams.get(&*suggestion_split.term) {
                                    Some(&bigram_frequency) => {
                                        // increase count, if split
                                        // corrections are part of or
                                        // identical to input single term
                                        // correction exists
                                        if !suggestions.is_empty() {
                                            let best_si = &suggestions[0];
                                            // # alternatively remove the
                                            // # single term from
                                            // # suggestion_split, but then
                                            // # other splittings could win
                                            if suggestion_split.term == term_list1[i] {
                                                // # make count bigger than
                                                // # count of single term
                                                // # correction
                                                cmp::max(bigram_frequency, best_si.count + 2)
                                            } else if suggestions1[0].term == best_si.term
                                                || suggestions2[0].term == best_si.term
                                            {
                                                // # make count bigger than
                                                // # count of single term
                                                // # correction
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
                                        // The Naive Bayes probability of
                                        // the word combination is the
                                        // product of the two word
                                        // probabilities: P(AB)=P(A)*P(B)
                                        // use it to estimate the frequency
                                        // count of the combination, which
                                        // then is used to rank/select the
                                        // best splitting variant
                                        min(
                                            self.bigram_min_count,
                                            ((suggestions1[0].count as f64)
                                                / (self.corpus_word_count as f64)
                                                * (suggestions2[0].count as f64))
                                                as i64,
                                        )
                                    }
                                };
                                suggestion_split.distance = distance2;
                                suggestion_split.count = count2;

                                //early termination of split
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
                        // NOTE: this effectively clamps si_count to a certain minimum value, which it can't go below
                        let si_count: f64 =
                            10f64 / ((10i64).saturating_pow(len(&term_list1[i]) as u32)) as f64;

                        si.term = term_list1[i].clone();
                        si.count = si_count as i64;
                        si.distance = edit_distance_max + 1;
                        suggestion_parts.push(si);
                    }
                } else {
                    let mut si = Suggestion::empty();
                    // NOTE: this effectively clamps si_count to a certain minimum value, which it can't go below
                    let si_count: f64 =
                        10f64 / ((10i64).saturating_pow(len(&term_list1[i]) as u32)) as f64;

                    si.term = term_list1[i].clone();
                    si.count = si_count as i64;
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

        suggestion.term = s.trim().to_string();
        suggestion.count = tmp_count as i64;
        suggestion.distance = damerau_levenshtein_osa(input, &suggestion.term, usize::MAX);

        vec![suggestion]
    }

    /// Divides a string into words by inserting missing spaces at the appropriate positions
    ///
    ///
    /// # Arguments
    ///
    /// * `input` - The word being segmented.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Examples
    ///
    /// ```
    /// use symspell_rs::{SymSpell, Verbosity};
    ///
    /// let mut symspell: SymSpell = SymSpell::new(2, 7, 1);
    /// symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    /// symspell.word_segmentation("itwas", 2);
    /// ```
    pub fn word_segmentation(&self, input: &str, max_edit_distance: i64) -> Composition {
        let asize = len(input);

        let mut ci: usize = 0;
        let mut compositions: Vec<Composition> = vec![Composition::empty(); asize];

        for j in 0..asize {
            let imax = min(asize - j, self.max_length as usize);
            for i in 1..=imax {
                let mut part = slice(input, j, j + i);

                let mut sep_len = 0;
                let mut top_ed: i64 = 0;

                let first_char = at(&part, 0).unwrap();
                if first_char.is_whitespace() {
                    part = remove(&part, 0);
                } else {
                    sep_len = 1;
                }

                top_ed += part.len() as i64;

                part = part.replace(" ", "");

                top_ed -= part.len() as i64;

                let results = self.lookup(&part, Verbosity::Top, max_edit_distance);

                let top_prob_log = if !results.is_empty() && results[0].distance == 0 {
                    (results[0].count as f64 / self.corpus_word_count as f64).log10()
                } else {
                    top_ed += part.len() as i64;
                    (10.0 / (self.corpus_word_count as f64 * 10.0f64.powf(part.len() as f64)))
                        .log10()
                };

                let di = (i + ci) % asize;
                // set values in first loop
                if j == 0 {
                    compositions[i - 1] = Composition {
                        segmented_string: part.to_owned(),
                        distance_sum: top_ed,
                        prob_log_sum: top_prob_log,
                    };
                } else if i as i64 == self.max_length
                    || (((compositions[ci].distance_sum + top_ed == compositions[di].distance_sum)
                        || (compositions[ci].distance_sum + sep_len + top_ed
                            == compositions[di].distance_sum))
                        && (compositions[di].prob_log_sum
                            < compositions[ci].prob_log_sum + top_prob_log))
                    || (compositions[ci].distance_sum + sep_len + top_ed
                        < compositions[di].distance_sum)
                {
                    compositions[di] = Composition {
                        segmented_string: format!("{} {}", compositions[ci].segmented_string, part),
                        distance_sum: compositions[ci].distance_sum + sep_len + top_ed,
                        prob_log_sum: compositions[ci].prob_log_sum + top_prob_log,
                    };
                }
            }
            if j != 0 {
                ci += 1;
            }
            ci = if ci == asize { 0 } else { ci };
        }
        compositions[ci].to_owned()
    }

    fn delete_in_suggestion_prefix(
        &self,
        delete: &str,
        delete_len: i64,
        suggestion: &str,
        suggestion_len: i64,
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

    fn create_dictionary_entry<K>(&mut self, key: K, count: i64) -> bool
    where
        K: Clone + AsRef<str> + Into<String>,
    {
        if count < self.count_threshold {
            return false;
        }

        let key_clone = key.clone().into().into_boxed_str();

        match self.words.get(key.as_ref()) {
            Some(i) => {
                let updated_count = if i64::MAX - i > count {
                    i + count
                } else {
                    i64::MAX
                };
                self.words.insert(key_clone, updated_count);
                return false;
            }
            None => {
                self.words.insert(key_clone, count);
            }
        }

        let key_len = len(key.as_ref());

        if key_len as i64 > self.max_length {
            self.max_length = key_len as i64;
        }

        let edits = self.edits_prefix(key.as_ref());

        for delete in edits {
            let delete_hash = hash64(delete.as_bytes());

            self.deletes
                .entry(delete_hash)
                .and_modify(|e| e.push(key.clone().into().into_boxed_str()))
                .or_insert_with(|| vec![key.clone().into().into_boxed_str()]);
        }

        true
    }

    fn edits_prefix(&self, key: &str) -> AHashSet<String> {
        let mut hash_set = AHashSet::new();

        let key_len = len(key) as i64;

        if key_len <= self.max_dictionary_edit_distance {
            hash_set.insert("".to_string());
        }

        if key_len > self.prefix_length {
            let shortened_key = slice(key, 0, self.prefix_length as usize);
            hash_set.insert(shortened_key.clone());
            self.edits(&shortened_key, 0, &mut hash_set);
        } else {
            hash_set.insert(key.to_string());
            self.edits(key, 0, &mut hash_set);
        };

        hash_set
    }

    fn edits(&self, word: &str, edit_distance: i64, delete_words: &mut AHashSet<String>) {
        let edit_distance = edit_distance + 1;
        let word_len = len(word);

        if word_len > 1 {
            for i in 0..word_len {
                let delete = remove(word, i);

                if !delete_words.contains(&delete) {
                    delete_words.insert(delete.clone());

                    if edit_distance < self.max_dictionary_edit_distance {
                        self.edits(&delete, edit_distance, delete_words);
                    }
                }
            }
        }
    }

    fn has_different_suffix(
        &self,
        max_edit_distance: i64,
        input: &str,
        input_len: i64,
        candidate_len: i64,
        suggestion: &str,
        suggestion_len: i64,
    ) -> bool {
        // handles the shortcircuit of min_distance
        // assignment when first boolean expression
        // evaluates to false
        let min = if self.prefix_length - max_edit_distance == candidate_len {
            min(input_len, suggestion_len) - self.prefix_length
        } else {
            0
        };

        (self.prefix_length - max_edit_distance == candidate_len)
            && (((min - self.prefix_length) > 1)
                && (suffix(input, (input_len + 1 - min) as usize)
                    != suffix(suggestion, (suggestion_len + 1 - min) as usize)))
            || ((min > 0)
                && (at(input, (input_len - min) as isize)
                    != at(suggestion, (suggestion_len - min) as isize))
                && ((at(input, (input_len - min - 1) as isize)
                    != at(suggestion, (suggestion_len - min) as isize))
                    || (at(input, (input_len - min) as isize)
                        != at(suggestion, (suggestion_len - min - 1) as isize))))
    }

    /// Parse a string into words, splitting at non-alphanumeric characters except for '_'.
    pub fn parse_words(&self, text: &str) -> Vec<String> {
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
            non_unique_terms_line
                .push(text_normalized[start_pos..text_normalized.len()].to_string());
        }

        non_unique_terms_line
    }
}

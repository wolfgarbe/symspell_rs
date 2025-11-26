/*!

Spelling correction & Fuzzy search based on Symmetric Delete spelling correction algorithm.

#### Usage of SymSpell Library

Single word spelling correction
```rust
use symspell_rs::{SymSpell, Verbosity};
use std::path::Path;

let max_edit_distance_dictionary = 2; //maximum edit distance per dictionary precalculation
let mut symspell: SymSpell = SymSpell::new(max_edit_distance_dictionary,None, 7, 1);

// single term dictionary
let term_index = 0; //column of the term in the dictionary text file
let count_index = 1; //column of the term frequency in the dictionary text file
symspell.load_dictionary(Path::new("data/frequency_dictionary_en_82_765.txt"), term_index, count_index, " ");

//lookup suggestions for single-word input strings
let input_term = "hous";
let suggestion_verbosity = Verbosity::Closest;//Top, Closest, All
let max_edit_distance_lookup = 1; //max edit distance per lookup (maxEditDistanceLookup<=maxEditDistanceDictionary)
let suggestions = symspell.lookup(input_term, suggestion_verbosity, max_edit_distance_lookup,None,None,false);
//display suggestions, edit distance and term frequency
println!("{:?}", suggestions);
```

Compound aware multi-word spelling correction
```rust
use symspell_rs::{SymSpell, Verbosity};
use std::path::Path;

let max_edit_distance_dictionary = 2; //maximum edit distance per dictionary precalculation
let mut symspell = SymSpell::new(max_edit_distance_dictionary, None,7, 1);

// single term dictionary
let term_index = 0; //column of the term in the dictionary text file
let count_index = 1; //column of the term frequency in the dictionary text file
symspell.load_dictionary(Path::new("data/frequency_dictionary_en_82_765.txt"), term_index, count_index, " ");
// bigram dictionary
symspell.load_bigram_dictionary(Path::new("data/frequency_bigramdictionary_en_243_342.txt"),0,2, " ",
);

//lookup suggestions for multi-word input strings (supports compound splitting & merging)
let input_sentence = "whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixtgrade and ins pired him";
let max_edit_distance_lookup = 2; //max edit distance per lookup (per single word, not per whole input string)
let compound_suggestions = symspell.lookup_compound(input_sentence, max_edit_distance_lookup,false);
//display suggestions, edit distance and term frequency
println!("{:?}", compound_suggestions);
```

Word Segmentation of noisy text
```rust
use symspell_rs::{SymSpell, Verbosity};
use std::path::Path;

let max_edit_distance_dictionary = 0; //maximum edit distance per dictionary precalculation
let mut symspell = SymSpell::new(max_edit_distance_dictionary, None,7, 1);

// single term dictionary
let term_index = 0; //column of the term in the dictionary text file
let count_index = 1; //column of the term frequency in the dictionary text file
symspell.load_dictionary(Path::new("data/frequency_dictionary_en_82_765.txt"), term_index, count_index, " ");

//word segmentation and correction for multi-word input strings with/without spaces
let input_sentence = "thequickbrownfoxjumpsoverthelazydog";
let max_edit_distance_lookup = 0;
let result = symspell.word_segmentation(input_sentence, max_edit_distance_lookup);
//display term and edit distance
println!("{:?}", result.segmented_string);
```

Word Segmentation of Chinese text
```rust
use symspell_rs::{SymSpell, Verbosity};
use std::path::Path;

let max_edit_distance_dictionary = 0; //maximum edit distance per dictionary precalculation
let mut symspell = SymSpell::new(max_edit_distance_dictionary,None, 7, 1);

// single term dictionary
let term_index = 0; //column of the term in the dictionary text file
let count_index = 1; //column of the term frequency in the dictionary text file
symspell.load_dictionary(Path::new("data/frequency_dictionary_zh_cn_349_045.txt"), term_index, count_index, " ");

//word segmentation and correction for multi-word input strings with/without spaces
let input_sentence = "部分居民生活水平";
let max_edit_distance_lookup = 0;
let result = symspell.word_segmentation(input_sentence, max_edit_distance_lookup);
//display term and edit distance
println!("{:?}", result.segmented_string);
```

*/

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

mod symspell;
mod test;
pub use symspell::{Suggestion, SymSpell, Verbosity, damerau_levenshtein_osa};

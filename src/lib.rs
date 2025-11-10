/*!

Spelling correction & Fuzzy search based on Symmetric Delete spelling correction algorithm.

#### Usage of SymSpell Library

```rust
use symspell_rs::{SymSpell, Verbosity};

let max_edit_distance_dictionary = 2; //maximum edit distance per dictionary precalculation
let mut symspell: SymSpell = SymSpell::new(max_edit_distance_dictionary, 7, 1);

let term_index:i64 = 0; //column of the term in the dictionary text file
let count_index:i64 = 1; //column of the term frequency in the dictionary text file

// single term dictionary
symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", term_index, count_index, " ");

//lookup suggestions for single-word input strings
let input_term = "hous";
let suggestion_verbosity = Verbosity::Closest;//Top, Closest, All
let max_edit_distance_lookup = 1; //max edit distance per lookup (maxEditDistanceLookup<=maxEditDistanceDictionary)
let suggestions = symspell.lookup(input_term, suggestion_verbosity, max_edit_distance_lookup);
//display suggestions, edit distance and term frequency
println!("{:?}", suggestions);

//###

let mut symspell = SymSpell::new(max_edit_distance_dictionary, 7, 1);

// single term dictionary
symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", term_index, count_index, " ");
// bigram dictionary
symspell.load_bigram_dictionary("data/frequency_bigramdictionary_en_243_342.txt",0,2, " ",
);

//lookup suggestions for multi-word input strings (supports compound splitting & merging)
let input_sentence = "whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixtgrade and ins pired him";
let max_edit_distance_lookup = 2; //max edit distance per lookup (per single word, not per whole input string)
let compound_suggestions = symspell.lookup_compound(input_sentence, max_edit_distance_lookup);
//display suggestions, edit distance and term frequency
println!("{:?}", compound_suggestions);

//###

let mut symspell = SymSpell::new(max_edit_distance_dictionary, 7, 1);

// single term dictionary
symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", term_index, count_index, " ");
// bigram dictionary
symspell.load_bigram_dictionary("data/frequency_bigramdictionary_en_243_342.txt",0,2, " ");

//word segmentation and correction for multi-word input strings with/without spaces
let input_sentence = "thequickbrownfoxjumpsoverthelazydog";
let max_edit_distance_lookup = 0;
let result = symspell.word_segmentation(input_sentence, max_edit_distance_lookup);
//display term and edit distance
println!("{:?}", result.segmented_string);
```

*/

mod symspell;
mod test;
pub use symspell::{SymSpell, Verbosity};

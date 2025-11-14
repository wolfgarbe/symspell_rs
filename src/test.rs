#[cfg(test)]
mod tests {
    use crate::{
        SymSpell, Verbosity,
        symspell::{transfer_case, unicode_normalization_form_kc},
    };

    #[test]
    fn test_lookup() {
        let edit_distance_max = 2;
        let mut symspell = SymSpell::new(edit_distance_max, 7, 1);
        symspell.load_dictionary("./data/frequency_dictionary_en_82_765.txt", 0, 1, " ");

        let typo = "hous";
        let correction = "house";
        let results = symspell.lookup(typo, Verbosity::Top, edit_distance_max, false);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(1, results[0].distance);
        assert_eq!(231310420, results[0].count);

        // case-insensitive lookup, but preserve original case in suggestion
        let typo = "Hous";
        let correction = "House";
        let results = symspell.lookup(typo, Verbosity::Top, edit_distance_max, true);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(1, results[0].distance);
        assert_eq!(231310420, results[0].count);
    }

    #[test]
    fn test_lookup_compound() {
        let edit_distance_max = 2;
        let mut symspell = SymSpell::new(edit_distance_max, 7, 1);
        symspell.load_dictionary("./data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
        symspell.load_bigram_dictionary(
            "./data/frequency_bigramdictionary_en_243_342.txt",
            0,
            2,
            " ",
        );

        let typo = "whereis th elove";
        let correction = "where is the love";
        let results = symspell.lookup_compound(typo, edit_distance_max, false);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(2, results[0].distance);
        assert_eq!(585, results[0].count);

        // preserve case
        let typo = "WHEREIS TH ELOVE";
        let correction = "WHERE IS THE LOVE";
        let results = symspell.lookup_compound(typo, edit_distance_max, true);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(2, results[0].distance);
        assert_eq!(585, results[0].count);

        let typo = "the bigjest playrs";
        let correction = "the biggest players";
        let results = symspell.lookup_compound(typo, edit_distance_max, false);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(2, results[0].distance);
        assert_eq!(34, results[0].count);

        let typo = "Can yu readthis";
        let correction = "can you read this";
        let results = symspell.lookup_compound(typo, edit_distance_max, false);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(2, results[0].distance);
        assert_eq!(1366, results[0].count);

        let typo = "whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixthgrade and ins pired him";
        let correction = "where is the love he had dated for much of the past who couldn't read in sixth grade and inspired him";
        let results = symspell.lookup_compound(typo, edit_distance_max, false);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(9, results[0].distance);
        assert_eq!(0, results[0].count);

        let typo = "in te dhird qarter oflast jear he hadlearned ofca sekretplan";
        let correction = "in the third quarter of last year he had learned of a secret plan";
        let results = symspell.lookup_compound(typo, edit_distance_max, false);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(9, results[0].distance);
        assert_eq!(0, results[0].count);

        let typo = "the bigjest playrs in te strogsommer film slatew ith plety of funn";
        let correction = "the biggest players in the strong summer film slate with plenty of fun";
        let results = symspell.lookup_compound(typo, edit_distance_max, false);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(9, results[0].distance);
        assert_eq!(0, results[0].count);

        let typo = "Can yu readthis messa ge despite thehorible sppelingmsitakes";
        let correction = "can you read this message despite the horrible spelling mistakes";
        let results = symspell.lookup_compound(typo, edit_distance_max, false);
        assert_eq!(1, results.len());
        assert_eq!(correction, results[0].term);
        assert_eq!(9, results[0].distance);
        assert_eq!(0, results[0].count);
    }

    #[test]
    fn test_word_segmentation() {
        let edit_distance_max = 1;
        let mut symspell = SymSpell::new(edit_distance_max, 7, 1);
        symspell.load_dictionary("./data/frequency_dictionary_en_82_765.txt", 0, 1, " ");

        let typo = "thequickbrownfoxjumpsoverthelazydog";
        let correction = "the quick brown fox jumps over the lazy dog";
        let result = symspell.word_segmentation(typo, 0);
        assert_eq!(correction, result.segmented_string);

        // works with upper case and preserves case
        let typo = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG";
        let correction = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG";
        let result = symspell.word_segmentation(typo, 0);
        assert_eq!(correction, result.segmented_string);

        // spell correct and preserve case for corrected term: THF -> THE
        let typo = "THFQUICKBROWNFOXJUMPSOVERTHELAZYDOG";
        let correction = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG";
        let result = symspell.word_segmentation(typo, 1);
        assert_eq!(correction, result.segmented_string);

        let typo = "itwasabrightcolddayinaprilandtheclockswerestrikingthirteen";
        let correction = "it was a bright cold day in april and the clocks were striking thirteen";
        let result = symspell.word_segmentation(typo, 0);
        assert_eq!(correction, result.segmented_string);

        let typo =
            "itwasthebestoftimesitwastheworstoftimesitwastheageofwisdomitwastheageoffoolishness";
        let correction = "it was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness";
        let result = symspell.word_segmentation(typo, 0);
        assert_eq!(correction, result.segmented_string);

        //keep punctuation or apostrophe adjacent to previous word, keep letter case
        let typo = "Idranktheglasses’contents,whichtastedofelderberries";
        let correction = "I drank the glasses’ contents, which tasted of elderberries";
        let result = symspell.word_segmentation(typo, 0);
        assert_eq!(correction, result.segmented_string);

        //keep punctuation or apostrophe adjacent to previous word, keep letter case
        let typo = "Idranktheglasses\'contents,whichtastedofelderberries";
        let correction = "I drank the glasses\' contents, which tasted of elderberries";
        let result = symspell.word_segmentation(typo, 0);
        assert_eq!(correction, result.segmented_string);
    }

    #[test]
    fn test_chinese_word_segmentation() {
        let edit_distance_max = 0;
        let mut symspell = SymSpell::new(edit_distance_max, 7, 1);
        symspell.load_dictionary("./data/frequency_dictionary_zh_cn_349_045.txt", 0, 1, " ");

        let typo = "部分居民生活水平";
        let correction = "部分 居民 生活 水平";
        let result = symspell.word_segmentation(typo, 0);
        assert_eq!(correction, result.segmented_string);
    }

    #[test]
    fn test_normalization() {
        let typo = "scientiﬁc";
        let correction = "scientific";
        let result = unicode_normalization_form_kc(typo);
        assert_eq!(correction, result);
    }

    #[test]
    fn test_transfer_case() {
        // transfer case with UTF8 characters, with shorter source
        let source = "LEG MOZE OZNACZAC LAKE W POBLIZU RZEKI";
        let target = "Łęg może oznaczać łąkę w pobliżu rzeki (łąka łęgowa)";
        let correction = "ŁĘG MOŻE OZNACZAĆ ŁĄKĘ W POBLIŻU RZEKI (ŁĄKA ŁĘGOWA)";
        let result = transfer_case(source, target);
        assert_eq!(correction, result);

        // transfer case with UTF8 characters, with shortcut
        let source = "LEG MOZE OZNACZAC LAKE W POBLIZU RZEKI (ŁĄKA ŁĘGOWA)";
        let target = "Łęg może oznaczać łąkę w pobliżu rzeki (łąka łęgowa)";
        let correction = "ŁĘG MOŻE OZNACZAĆ ŁĄKĘ W POBLIŻU RZEKI (ŁĄKA ŁĘGOWA)";
        let result = transfer_case(source, target);
        assert_eq!(correction, result);
    }
}

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use strsim::osa_distance;

//use symspell_old::*;

#[path = "../src/symspell.rs"]
mod symspell_new;

const DICTS: [&str; 3] = [
    "frequency_dictionary_en_30_000.txt",
    "frequency_dictionary_en_82_765.txt",
    "frequency_dictionary_en_500_000.txt",
];

fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("test_data")
        .join(name)
}

pub struct TestEntry {
    /// misspelled string,
    pub misspelled_string: String,
    /// correct string,
    pub correct_string: String,
    /// edit distance between the misspelled string and the correct string.
    pub edit_distance: usize,
}

fn main() {
    // Fixes a 85% performance drop caused by a faulty Windows 11 24H2 software update,
    // that changed the task scheduler behavior into under-utilizing the P-Cores over E-cores of Intel hybrid CPUs.
    // This is a workaround until Microsoft fixes the issue in a future update.
    // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass
    #[cfg(target_os = "windows")]
    unsafe {
        use winapi::um::{
            processthreadsapi::SetPriorityClass, winbase::ABOVE_NORMAL_PRIORITY_CLASS,
        };

        let process = winapi::um::processthreadsapi::GetCurrentProcess();
        SetPriorityClass(process, ABOVE_NORMAL_PRIORITY_CLASS);
    }

    //### load queries
    let noisy_query_file = data_path("noisy_query_en_1000.txt");
    let mut test_entries: Vec<TestEntry> = Vec::new();

    let file = File::open(noisy_query_file).unwrap();
    let sr = BufReader::new(file);

    for line in sr.lines() {
        let line_str = line.unwrap();

        let line_parts: Vec<&str> = line_str.split(' ').collect();
        if line_parts.len() >= 3 {
            let misspelled_string = line_parts[0].to_string();
            let correct_string = line_parts[1].to_string();
            let edit_distance = line_parts[2].parse::<usize>().unwrap();

            test_entries.push(TestEntry {
                misspelled_string: misspelled_string,
                correct_string: correct_string,
                edit_distance,
            });
        }
    }

    let count_threshold = 1;
    let term_index = 0; //column of the term in the dictionary text file
    let count_index = 1; //column of the term frequency in the dictionary text file
    let separator = " ";

    let query_count = test_entries.len() as u32;
    let loop_count = 100u32;
    let sleep_time = Duration::from_secs(10);

    let old_version = "v6.8.4";
    let current_version = concat!("v", env!("CARGO_PKG_VERSION"));

    //### warmup

    println!("--- Warmup query count: {}", query_count);

    let max_edit_distance = 1; //maximum edit distance per dictionary precalculation
    let prefix_length = 5;

    let mut new_symspell =
        symspell_new::SymSpell::new(max_edit_distance, None, prefix_length, count_threshold);
    let _ = new_symspell.load_dictionary(&data_path(DICTS[0]), term_index, count_index, separator);

    thread::sleep(sleep_time);
    let start_time = Instant::now();
    let mut suggestions_sum = 0;
    for _i in 0..loop_count {
        for test_entry in &test_entries {
            let suggestions = new_symspell.lookup(
                test_entry.misspelled_string.as_str(),
                symspell_new::Verbosity::Top,
                max_edit_distance,
                &None,
                None,
                false,
            );
            suggestions_sum += suggestions.len();
        }
    }
    let duration = start_time.elapsed();
    println!(
        "SymSpell {}  Warmup  prefix_length: {}  max_edit_distance: {} Top  lookup_latency: {:?}  total_suggestions: {}",
        current_version,
        prefix_length,
        max_edit_distance,
        duration / loop_count / query_count,
        suggestions_sum / loop_count as usize
    );

    //### benchmark loops

    for dict in DICTS {
        println!("Dictionary: {}", dict);

        for prefix_length in 5..=7 {
            for max_edit_distance in 1..=3 {
                //### dictionary load

                //old
                thread::sleep(sleep_time);
                let start_time = Instant::now();

                let mut old_symspell = symspell_old::SymSpell::new(
                    max_edit_distance,
                    None,
                    prefix_length,
                    count_threshold,
                );
                let _ = old_symspell.load_dictionary(
                    &data_path(dict),
                    term_index,
                    count_index,
                    separator,
                );
                let old_duration = start_time.elapsed();
                println!(
                    "SymSpell {}  prefix_length: {}  max_edit_distance: {} dictionary load time: {:?}  ratio 1.0  term_count: {}",
                    old_version,
                    prefix_length,
                    max_edit_distance,
                    old_duration,
                    old_symspell.get_dictionary_size()
                );

                //new
                thread::sleep(sleep_time);
                let start_time = Instant::now();

                let mut new_symspell = symspell_new::SymSpell::new(
                    max_edit_distance,
                    None,
                    prefix_length,
                    count_threshold,
                );
                let _ = new_symspell.load_dictionary(
                    &data_path(dict),
                    term_index,
                    count_index,
                    separator,
                );
                let new_duration = start_time.elapsed();
                println!(
                    "SymSpell {}  prefix_length: {}  max_edit_distance: {} dictionary load time: {:?}  ratio {:.2}  term_count: {}",
                    current_version,
                    prefix_length,
                    max_edit_distance,
                    new_duration,
                    old_duration.as_nanos() as f64 / new_duration.as_nanos() as f64,
                    new_symspell.get_dictionary_size()
                );

                //### lookup

                //### top

                //old

                thread::sleep(sleep_time);
                let start_time = Instant::now();
                let mut suggestions_sum = 0;
                for _i in 0..loop_count {
                    for test_entry in &test_entries {
                        let suggestions = old_symspell.lookup(
                            test_entry.misspelled_string.as_str(),
                            symspell_old::Verbosity::Top,
                            max_edit_distance,
                            &None,
                            None,
                            false,
                        );
                        suggestions_sum += suggestions.len();
                    }
                }
                let old_duration = start_time.elapsed();
                println!(
                    "SymSpell {}  prefix_length: {}  max_edit_distance: {} Top  lookup_latency: {:?}  ratio 1.0  total_suggestions: {}",
                    old_version,
                    prefix_length,
                    max_edit_distance,
                    old_duration / loop_count / query_count,
                    suggestions_sum / loop_count as usize
                );

                //new

                thread::sleep(sleep_time);
                let start_time = Instant::now();
                let mut suggestions_sum = 0;
                for _i in 0..loop_count {
                    for test_entry in &test_entries {
                        let suggestions = new_symspell.lookup(
                            test_entry.misspelled_string.as_str(),
                            symspell_new::Verbosity::Top,
                            max_edit_distance,
                            &None,
                            None,
                            false,
                        );
                        suggestions_sum += suggestions.len();
                    }
                }
                let new_duration = start_time.elapsed();
                println!(
                    "SymSpell {}  prefix_length: {}  max_edit_distance: {} Top  lookup_latency: {:?}  ratio {:.2}  total_suggestions: {}",
                    current_version,
                    prefix_length,
                    max_edit_distance,
                    new_duration / loop_count / query_count,
                    old_duration.as_nanos() as f64 / new_duration.as_nanos() as f64,
                    suggestions_sum / loop_count as usize
                );

                //### closest

                //old

                thread::sleep(sleep_time);
                let start_time = Instant::now();
                let mut suggestions_sum = 0;
                for _i in 0..loop_count {
                    for test_entry in &test_entries {
                        let suggestions = old_symspell.lookup(
                            test_entry.misspelled_string.as_str(),
                            symspell_old::Verbosity::Closest,
                            max_edit_distance,
                            &None,
                            None,
                            false,
                        );
                        suggestions_sum += suggestions.len();
                    }
                }
                let old_duration = start_time.elapsed();
                println!(
                    "SymSpell {}  prefix_length: {}  max_edit_distance: {} Closest  lookup_latency: {:?}  ratio 1.0  total_suggestions: {}",
                    old_version,
                    prefix_length,
                    max_edit_distance,
                    old_duration / loop_count / query_count,
                    suggestions_sum / loop_count as usize
                );

                //new

                thread::sleep(sleep_time);
                let start_time = Instant::now();
                let mut suggestions_sum = 0;
                for _i in 0..loop_count {
                    for test_entry in &test_entries {
                        let suggestions = new_symspell.lookup(
                            test_entry.misspelled_string.as_str(),
                            symspell_new::Verbosity::Closest,
                            max_edit_distance,
                            &None,
                            None,
                            false,
                        );
                        suggestions_sum += suggestions.len();
                    }
                }
                let new_duration = start_time.elapsed();
                println!(
                    "SymSpell {}  prefix_length: {}  max_edit_distance: {} Closest  lookup_latency: {:?}  ratio {:.2}  total_suggestions: {}",
                    current_version,
                    prefix_length,
                    max_edit_distance,
                    new_duration / loop_count / query_count,
                    old_duration.as_nanos() as f64 / new_duration.as_nanos() as f64,
                    suggestions_sum / loop_count as usize
                );

                //### all

                //old

                thread::sleep(sleep_time);
                let start_time = Instant::now();
                let mut suggestions_sum = 0;
                for _i in 0..loop_count {
                    for test_entry in &test_entries {
                        let suggestions = old_symspell.lookup(
                            test_entry.misspelled_string.as_str(),
                            symspell_old::Verbosity::All,
                            max_edit_distance,
                            &None,
                            None,
                            false,
                        );
                        suggestions_sum += suggestions.len();
                    }
                }
                let old_duration = start_time.elapsed();
                println!(
                    "SymSpell {}  prefix_length: {}  max_edit_distance: {} All  lookup_latency: {:?}  ratio 1.0  total_suggestions: {}",
                    old_version,
                    prefix_length,
                    max_edit_distance,
                    old_duration / loop_count / query_count,
                    suggestions_sum / loop_count as usize
                );

                //new

                thread::sleep(sleep_time);
                let start_time = Instant::now();
                let mut suggestions_sum = 0;
                for _i in 0..loop_count {
                    for test_entry in &test_entries {
                        let suggestions = new_symspell.lookup(
                            test_entry.misspelled_string.as_str(),
                            symspell_new::Verbosity::All,
                            max_edit_distance,
                            &None,
                            None,
                            false,
                        );
                        suggestions_sum += suggestions.len();
                    }
                }
                let new_duration = start_time.elapsed();
                println!(
                    "SymSpell {}  prefix_length: {}  max_edit_distance: {} All  lookup_latency: {:?}  ratio {:.2}  total_suggestions: {}",
                    current_version,
                    prefix_length,
                    max_edit_distance,
                    new_duration / loop_count / query_count,
                    old_duration.as_nanos() as f64 / new_duration.as_nanos() as f64,
                    suggestions_sum / loop_count as usize
                );
            }
        }
    }

    //### Damerau-Levenshtein OSA benchmark

    let loop_count = 1000;

    //warmup
    thread::sleep(sleep_time);

    let max_edit_distance = 1;
    let start_time = Instant::now();
    let mut edit_distance_sum = 0;
    for _i in 0..loop_count {
        for test_entry in &test_entries {
            let result = symspell_old::damerau_levenshtein_osa(
                test_entry.misspelled_string.as_str(),
                test_entry.correct_string.as_str(),
                max_edit_distance,
            );

            if let Some(edit_distance) = result {
                edit_distance_sum += edit_distance;
            }
        }
    }
    let old_duration = start_time.elapsed();
    println!(
        "Damerau-Levenshtein OSA warmup  SymSpell {}  max_edit_distance: {}  latency: {:?}  total_edit_distance: {}",
        old_version,
        max_edit_distance,
        old_duration / loop_count / query_count,
        edit_distance_sum / loop_count as usize
    );

    for max_edit_distance in 1..=3 {
        // symspell_old::damerau_levenshtein_osa

        thread::sleep(sleep_time);

        let start_time = Instant::now();
        let mut edit_distance_sum = 0;
        for _i in 0..loop_count {
            for test_entry in &test_entries {
                let result = symspell_old::damerau_levenshtein_osa(
                    test_entry.misspelled_string.as_str(),
                    test_entry.correct_string.as_str(),
                    max_edit_distance,
                );

                if let Some(edit_distance) = result {
                    edit_distance_sum += edit_distance;
                }
            }
        }
        let old_duration = start_time.elapsed();
        println!(
            "Damerau-Levenshtein OSA  SymSpell {}  max_edit_distance: {}  latency: {:?}  total_edit_distance: {}",
            old_version,
            max_edit_distance,
            old_duration / loop_count / query_count,
            edit_distance_sum / loop_count as usize
        );

        // symspell_new::damerau_levenshtein_osa_fallback (unlimited string length) vs. symspell_old::damerau_levenshtein_osa

        thread::sleep(sleep_time);

        let start_time = Instant::now();
        let mut edit_distance_sum = 0;
        for _i in 0..loop_count {
            for test_entry in &test_entries {
                let result = symspell_new::damerau_levenshtein_osa_fallback(
                    test_entry.misspelled_string.as_str(),
                    test_entry.correct_string.as_str(),
                    max_edit_distance,
                );

                if let Some(edit_distance) = result {
                    edit_distance_sum += edit_distance;
                }
            }
        }
        let new_duration = start_time.elapsed();
        println!(
            "Damerau-Levenshtein OSA  SymSpell {} fallback  max_edit_distance: {}  latency: {:?}  ( {:.2}x faster than SymSpell {} )  total_edit_distance: {}",
            current_version,
            max_edit_distance,
            new_duration / loop_count / query_count,
            old_duration.as_nanos() as f64 / new_duration.as_nanos() as f64,
            old_version,
            edit_distance_sum / loop_count as usize
        );

        // symspell_new::damerau_levenshtein_osa (<= 64 chars) vs. symspell_old::damerau_levenshtein_osa

        thread::sleep(sleep_time);

        let start_time = Instant::now();
        let mut edit_distance_sum = 0;
        for _i in 0..loop_count {
            for test_entry in &test_entries {
                let result = symspell_new::damerau_levenshtein_osa(
                    test_entry.misspelled_string.as_str(),
                    test_entry.correct_string.as_str(),
                    max_edit_distance,
                );

                if let Some(edit_distance) = result {
                    edit_distance_sum += edit_distance;
                }
            }
        }
        let new_duration = start_time.elapsed();
        println!(
            "Damerau-Levenshtein OSA  SymSpell {}  max_edit_distance: {}  latency: {:?}  ( {:.2}x faster than SymSpell {} )  total_edit_distance: {}",
            current_version,
            max_edit_distance,
            new_duration / loop_count / query_count,
            old_duration.as_nanos() as f64 / new_duration.as_nanos() as f64,
            old_version,
            edit_distance_sum / loop_count as usize
        );

        // strsim.osa_distance vs. symspell_new::damerau_levenshtein_osa

        thread::sleep(sleep_time);

        let start_time = Instant::now();
        let mut edit_distance_sum = 0;
        for _i in 0..loop_count {
            for test_entry in &test_entries {
                let result =
                    osa_distance(&test_entry.misspelled_string, &test_entry.correct_string);
                if result <= max_edit_distance {
                    edit_distance_sum += result;
                }
            }
        }
        let strsim_duration = start_time.elapsed();
        println!(
            "Damerau-Levenshtein OSA  SymSpell {}  max_edit_distance: {}  latency: {:?}  ( {:.2}x faster than strsim v0.11.1 )  total_edit_distance: {}",
            current_version,
            max_edit_distance,
            strsim_duration / loop_count / query_count,
            strsim_duration.as_nanos() as f64 / new_duration.as_nanos() as f64,
            edit_distance_sum / loop_count as usize
        );
    }
}

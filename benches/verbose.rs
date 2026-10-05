use divan::Bencher;
use divan::counter::ItemsCount;
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Allocator wrapper tracking current and peak allocated bytes.
struct PeakAlloc;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CURRENT.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let np = unsafe { System.realloc(p, l, new_size) };
        if !np.is_null() {
            if new_size >= l.size() {
                let now = CURRENT.fetch_add(new_size - l.size(), Relaxed) + new_size - l.size();
                PEAK.fetch_max(now, Relaxed);
            } else {
                CURRENT.fetch_sub(l.size() - new_size, Relaxed);
            }
        }
        np
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;

const DICTS: [&str; 3] = [
    "frequency_dictionary_en_30_000.txt",
    "frequency_dictionary_en_82_765.txt",
    "frequency_dictionary_en_500_000.txt",
];

#[derive(Clone, Copy, Debug)]
struct Config {
    dict: &'static str,
    max_edit_distance: usize,
    prefix_length: usize,
    verbosity: &'static str,
    version: &'static str,
}

const CURRENT_VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));
const OLD_VERSION: &str = "v6.8.4";

/// Same API in both crates; `$krate` is the crate to benchmark.
macro_rules! engine {
    ($m:ident, $krate:ident) => {
        mod $m {
            use super::{Config, data_path};
            pub use $krate::SymSpell;
            use $krate::Verbosity;

            pub fn build(c: &Config) -> SymSpell {
                let mut s = SymSpell::new(c.max_edit_distance, None, c.prefix_length, 1);
                s.load_dictionary(&data_path(c.dict), 0, 1, " ").unwrap();
                s
            }

            pub fn run(s: &SymSpell, qs: &[String], c: &Config) -> usize {
                let mut n = 0;
                for q in qs {
                    let v = match c.verbosity {
                        "Top" => Verbosity::Top,
                        "Closest" => Verbosity::Closest,
                        _ => Verbosity::All,
                    };
                    n += s
                        .lookup(q, v, c.max_edit_distance, &None, None, false)
                        .len();
                }
                n
            }
        }
    };
}
engine!(current, symspell_rs);
engine!(old, symspell_old);

fn configs() -> Vec<Config> {
    let mut v = Vec::new();
    for dict in DICTS {
        for prefix_length in [5, 6, 7] {
            for max_edit_distance in [1, 2, 3] {
                for verbosity in ["Top", "Closest", "All"] {
                    for version in [CURRENT_VERSION, OLD_VERSION] {
                        v.push(Config {
                            dict,
                            max_edit_distance,
                            prefix_length,
                            verbosity,
                            version,
                        });
                    }
                }
            }
        }
    }
    v
}

fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmark/test_data")
        .join(name)
}

fn queries() -> Arc<Vec<String>> {
    static Q: Mutex<Option<Arc<Vec<String>>>> = Mutex::new(None);
    let mut g = Q.lock().unwrap();
    g.get_or_insert_with(|| {
        let text = std::fs::read_to_string(data_path("noisy_query_en_1000.txt")).unwrap();
        Arc::new(
            text.lines()
                .filter_map(|l| l.split(' ').next())
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect(),
        )
    })
    .clone()
}

struct Loaded {
    key: (&'static str, usize, usize),
    current: Arc<current::SymSpell>,
    old: Arc<old::SymSpell>,
}

/// Builds one dictionary and returns it with its build time and memory figures.
fn build_measured<T>(label: &str, c: &Config, build: impl FnOnce(&Config) -> T) -> T {
    let before = CURRENT.load(Relaxed);
    PEAK.store(before, Relaxed);
    let t = Instant::now();
    let s = build(c);
    eprintln!(
        "[dictionary {} {} prefix_length={} max_ed={}] build {:.1?}, resident {:.1} MiB, build peak {:.1} MiB",
        label,
        c.dict,
        c.prefix_length,
        c.max_edit_distance,
        t.elapsed(),
        mib(CURRENT.load(Relaxed) - before),
        mib(PEAK.load(Relaxed) - before),
    );
    s
}

/// Keeps only one dictionary pair alive at a time (configs are ordered dictionary-major).
fn dictionaries(c: &Config) -> (Arc<current::SymSpell>, Arc<old::SymSpell>) {
    static CACHE: Mutex<Option<Loaded>> = Mutex::new(None);
    let mut g = CACHE.lock().unwrap();
    let key = (c.dict, c.prefix_length, c.max_edit_distance);
    if let Some(l) = g.as_ref() {
        if l.key == key {
            return (l.current.clone(), l.old.clone());
        }
    }
    *g = None;

    let cur = Arc::new(build_measured(CURRENT_VERSION, c, current::build));
    let old = Arc::new(build_measured(OLD_VERSION, c, old::build));
    *g = Some(Loaded {
        key,
        current: cur.clone(),
        old: old.clone(),
    });
    (cur, old)
}

fn mib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

struct Stats {
    avg: Duration,
    peak: usize,
}

/// Stats of the current version, kept so the v6.8.4 run (next in order) can print the comparison.
static CURRENT_STATS: Mutex<Option<Stats>> = Mutex::new(None);

#[divan::bench(args = configs(), sample_count = 10)]
fn lookup(bencher: Bencher, c: &Config) {
    let (cur, old) = dictionaries(c);
    let qs = queries();
    let is_current = c.version == CURRENT_VERSION;
    let run = |qs: &[String]| {
        if is_current {
            current::run(&cur, qs, c)
        } else {
            old::run(&old, qs, c)
        }
    };

    // one measured pass: average latency and peak lookup allocation above the resident baseline
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t = Instant::now();
    std::hint::black_box(run(&qs));
    let stats = Stats {
        avg: t.elapsed() / qs.len() as u32,
        peak: PEAK.load(Relaxed) - base,
    };
    eprint!(
        "[{} prefix={} ed={} {} {}] avg latency {:.2?}/lookup, peak lookup alloc {} B",
        c.dict, c.prefix_length, c.max_edit_distance, c.verbosity, c.version, stats.avg, stats.peak,
    );
    if is_current {
        eprintln!();
        *CURRENT_STATS.lock().unwrap() = Some(stats);
    } else if let Some(cs) = CURRENT_STATS.lock().unwrap().as_ref() {
        eprintln!(
            "\n    => current vs {}: latency {:.2}x faster, peak alloc {:.2}x of {}",
            OLD_VERSION,
            stats.avg.as_secs_f64() / cs.avg.as_secs_f64(),
            cs.peak as f64 / stats.peak.max(1) as f64,
            OLD_VERSION,
        );
    }

    bencher
        .counter(ItemsCount::new(qs.len()))
        .bench_local(|| run(&qs));
}

fn main() {
    divan::main();
}

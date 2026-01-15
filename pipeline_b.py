# ===========================
# Pipeline B — Statistical Feature Extraction
# This pipeline treats the ELF file as a raw byte sequence and extracts
# statistical patterns that may indicate packing, encryption, or obfuscation.
# Extracts statistical features from ELF file:
# - Entropy (global and windowed)
# - N-grams (byte patterns)
# - Byte distribution statistics
# ===========================
"""
pipeline_b.py

This module provides functions to extract statistical features from ELF binaries,
treating the file primarily as a raw byte stream. The features include entropy
measures (global and windowed), coarse byte-distribution histograms, n-gram
statistics, header-specific features, and section entropy computed via pyelftools.

The goal of the added comments and docstrings in this version is to explain
every line or small block so that developers can reason about the implementation
in full detail.
"""

# Standard library imports
import math  # math functions (log2, sqrt)
from collections import Counter  # efficient counting of hashable objects (bytes or tuples)
import io  # BytesIO for treating bytes as file-like object

# Third-party import: pyelftools to parse ELF structures
# elftools may raise ImportError at runtime if not installed.
from elftools.elf.elffile import ELFFile

# ===========================
# Compute Shannon entropy of a byte sequence.
# High entropy may indicate compressed or encrypted regions.
# ===========================
def entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy of a byte sequence.

    Parameters:
    - data (bytes): Raw bytes to compute entropy on. Typically the content of an ELF
      section or the whole file.

    Returns:
    - float: Shannon entropy in bits per byte. If data is empty returns 0.0.

    Explanation:
    - Shannon entropy H = - sum(p_i * log2(p_i)) where p_i is the probability of each
      distinct byte value in the sequence. This measures unpredictability; higher
      values indicate more randomness/compression/encryption.
    """
    # If there is no data, entropy is 0 by definition (nothing to be random).
    if not data:
        return 0.0

    # Count occurrences of every distinct byte value (0-255) in the data.
    counts = Counter(data)

    # Total number of bytes in sequence (used to compute probabilities).
    total = len(data)

    # Calculate sum of -p*log2(p) across byte frequencies
    # c/total is the probability of that byte; guard against zero not needed because Counter excludes zero counts.
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ===========================
# Compute entropy over sliding windows across the file.
# This captures local entropy spikes instead of only global averages.
# ===========================
def windowed_entropy(data: bytes, window_size: int = 256, step: int = 128) -> dict:
    """
    Calculate entropy statistics over sliding windows.

    Parameters:
    - data (bytes): The raw byte sequence to scan with windows.
    - window_size (int): Size of each window in bytes. Default 256.
    - step (int): Step between window start positions in bytes. Default 128 (50% overlap).

    Returns:
    - dict: Dictionary with keys:
        'entropy_min'  : minimum window entropy observed
        'entropy_max'  : maximum window entropy observed
        'entropy_mean' : mean entropy across windows
        'entropy_std'  : standard deviation of window entropies

    Explanation:
    - If the file is smaller than the window, we compute entropy on the entire file
      and return that value for min/max/mean with std=0.
    - Sliding windows capture local regions of high/low entropy (packed regions, embedded resources).
    """
    # If data length is smaller than a full window, compute entropy for the whole data once.
    if len(data) < window_size:
        e = entropy(data)  # reuse entropy() to compute the scalar
        return {
            'entropy_min': e,  # minimum is the single value
            'entropy_max': e,  # maximum is the single value
            'entropy_mean': e,  # mean is the single value
            'entropy_std': 0.0,  # standard deviation is zero (single sample)
        }

    # List to store entropy of each window
    entropies = []

    # Iterate window start positions from 0 to last possible start, stepping by `step`.
    # Range stop: len(data) - window_size + 1 ensures last window is fully inside data.
    for i in range(0, len(data) - window_size + 1, step):
        # Extract the window slice
        window = data[i:i + window_size]
        # Compute entropy for this window and append
        entropies.append(entropy(window))

    # If no windows were produced (edge-case), return zeros to indicate no information.
    if not entropies:
        return {
            'entropy_min': 0.0,
            'entropy_max': 0.0,
            'entropy_mean': 0.0,
            'entropy_std': 0.0,
        }

    # Compute mean entropy of windows
    mean = sum(entropies) / len(entropies)

    # Compute variance as average squared deviation (population variance).
    variance = sum((e - mean) ** 2 for e in entropies) / len(entropies)

    # Return aggregated statistics: min, max, mean, std (sqrt of variance)
    return {
        'entropy_min': min(entropies),
        'entropy_max': max(entropies),
        'entropy_mean': mean,
        'entropy_std': math.sqrt(variance),
    }


# ===========================
# Build coarse byte-distribution features using 16 bins.
# Also track special byte categories (null, printable, high bytes).
# ===========================
def byte_histogram(data: bytes) -> dict:
    """
    Calculate byte frequency distribution features.

    Parameters:
    - data (bytes): Raw bytes of the file.

    Returns:
    - dict: Contains:
        - 'byte_freq_bin_0' ... 'byte_freq_bin_15' : normalized frequency of bytes
          falling into 16 coarse bins (each bin spans 16 byte values).
        - 'null_byte_freq' : frequency of 0x00 byte
        - 'printable_freq' : frequency of ASCII printable bytes (0x20 to 0x7E inclusive)
        - 'high_byte_freq'  : frequency of bytes >= 0x80 (non-ASCII high bytes)

    Explanation:
    - Binning reduces dimensionality from 256 to 16 while preserving coarse distribution.
    - The special counts help detect padding (nulls), text-heavy blobs (printable),
      and potential encrypted or binary-heavy regions (high bytes).
    """
    # If empty data, return zeros for the 16 bins to keep consistent feature names.
    if not data:
        return {f'byte_freq_{i}': 0.0 for i in range(16)}

    # Count every byte value in the data (0..255)
    counts = Counter(data)

    # Total number of bytes for normalization
    total = len(data)

    # Build the 16-bin histogram: each bin represents 16 consecutive byte values.
    features = {}
    for bin_idx in range(16):
        # Calculate start and end of this bin (inclusive of start, exclusive of end)
        start = bin_idx * 16
        end = (bin_idx + 1) * 16
        # Sum counts for bytes in that bin range
        bin_count = sum(counts.get(b, 0) for b in range(start, end))
        # Normalize to frequency and store with a descriptive key
        features[f'byte_freq_bin_{bin_idx}'] = bin_count / total

    # Additional individual features:
    # Frequency of null bytes (0x00)
    features['null_byte_freq'] = counts.get(0, 0) / total
    # Frequency of printable ASCII characters (0x20 to 0x7E)
    features['printable_freq'] = sum(counts.get(b, 0) for b in range(32, 127)) / total
    # Frequency of high bytes (>=0x80), often indicative of binary/compressed/encrypted data
    features['high_byte_freq'] = sum(counts.get(b, 0) for b in range(128, 256)) / total

    return features


# ===========================
# Extract N-gram statistics from byte stream (n = 2,3,4).
# These features capture short byte-pattern repetitions.
# ===========================
def ngram_features(data: bytes, n_values: list = [2, 3, 4]) -> dict:
    """
    Extract N-gram features from byte sequence.

    Parameters:
    - data (bytes): Raw bytes of the file.
    - n_values (list[int]): List of n sizes to compute (e.g., [2,3,4]).

    Returns:
    - dict: For each n in n_values, includes:
        - 'ngram_{n}_unique' : number of unique n-grams
        - 'ngram_{n}_entropy' : Shannon entropy of n-gram distribution (bits per n-gram)
        - 'ngram_{n}_most_common_freq' : relative frequency of the most common n-gram
        - 'ngram_{n}_diversity_50' : fraction of unique n-grams needed to cover 50% of occurrences

    Explanation:
    - N-grams are contiguous byte tuples of length n. They expose short repeated
      patterns (instruction sequences, repeated constants, markers), which can be
      useful to detect packing or code reuse.
    """
    features = {}  # dictionary to accumulate computed features

    # Loop over desired n values (e.g., 2, 3, 4)
    for n in n_values:
        # If data length is shorter than n, we cannot form any n-gram
        if len(data) < n:
            # Provide default zeroed features for consistency
            features[f'ngram_{n}_unique'] = 0
            features[f'ngram_{n}_entropy'] = 0.0
            features[f'ngram_{n}_most_common_freq'] = 0.0
            continue  # proceed to next n

        # Build list of n-grams as tuples (immutable, hashable for Counter).
        # There are len(data) - n + 1 overlapping n-grams.
        ngrams = [tuple(data[i:i + n]) for i in range(len(data) - n + 1)]
        # Count occurrences of each n-gram
        ngram_counts = Counter(ngrams)

        # Number of unique n-grams
        features[f'ngram_{n}_unique'] = len(ngram_counts)

        # Compute entropy for n-gram distribution:
        total = len(ngrams)  # total n-gram occurrences (for probability normalization)
        # Sum -p*log2(p) over n-gram counts
        ngram_ent = -sum((c / total) * math.log2(c / total) for c in ngram_counts.values())
        features[f'ngram_{n}_entropy'] = ngram_ent

        # Frequency of the most common n-gram (normalized)
        most_common_count = ngram_counts.most_common(1)[0][1] if ngram_counts else 0
        features[f'ngram_{n}_most_common_freq'] = most_common_count / total if total > 0 else 0.0

        # Compute "diversity_50": fraction of unique n-grams required to account for 50% of total occurrences.
        # Sort counts in descending order and accumulate until reaching >= 50% of total.
        sorted_counts = sorted(ngram_counts.values(), reverse=True)
        cumsum = 0
        for i, count in enumerate(sorted_counts):
            cumsum += count
            if cumsum >= total * 0.5:
                # (i + 1) is number of top n-grams needed; divide by number of unique n-grams to get fraction.
                features[f'ngram_{n}_diversity_50'] = (i + 1) / len(sorted_counts) if sorted_counts else 0.0
                break
        else:
            # If not broken (shouldn't happen), set diversity to 1.0 (all n-grams required)
            features[f'ngram_{n}_diversity_50'] = 1.0

    return features


# ===========================
# Analyze only the ELF header area (first ~1KB).
# Useful for detecting abnormal or obfuscated headers.
# ===========================
def header_features(data: bytes) -> dict:
    """
    Extract features from ELF header region.

    Parameters:
    - data (bytes): Raw bytes of the file.

    Returns:
    - dict: Contains:
        - 'header_entropy'     : entropy of the first ~1024 bytes (header region)
        - 'header_null_freq'   : fraction of null bytes in the header region
        - 'elf_header_entropy' : entropy of the first 64 bytes (ELF header)

    Explanation:
    - The ELF header and initial bytes often contain identifiable structure.
      Obfuscation or packing sometimes alters or replaces expected header fields,
      so these measures can be useful signals.
    """
    # Only look at the initial 1024 bytes to capture header and immediate metadata
    header_data = data[:1024] if len(data) > 1024 else data

    # Initialize features dictionary
    features = {}
    # Entropy on the chosen header region
    features['header_entropy'] = entropy(header_data)
    # Fraction of zero bytes in the header region; avoid division by zero by checking header_data
    features['header_null_freq'] = header_data.count(0) / len(header_data) if header_data else 0.0

    # ELF header is typically the first 64 bytes for 64-bit ELF (less for smaller files).
    elf_header = data[:64] if len(data) >= 64 else data
    features['elf_header_entropy'] = entropy(elf_header)

    return features


# ===========================
# Compute entropy of specific ELF sections (.text, .data, .rodata) and size of .bss.
# This partially reintroduces structural awareness into a statistical pipeline.
# ===========================
def section_entropy_features(data: bytes) -> dict:
    """
    Calculate entropy of each section type using pyelftools.

    Parameters:
    - data (bytes): Raw bytes of the ELF file.

    Returns:
    - dict: Contains:
        - 'text_section_entropy'   : entropy for .text section (code) if present
        - 'data_section_entropy'   : entropy for .data section if present
        - 'rodata_section_entropy' : entropy for .rodata (read-only data) if present
        - 'bss_section_size'       : size in bytes of the .bss section (SHT_NOBITS)

    Explanation:
    - The function attempts to parse the ELF structure. For each section we check
      the standard names and compute entropy on the section contents (unless the
      section is NOBITS, like .bss, which has no stored bytes).
    - Any parsing errors are silently ignored and default values are returned.
      This behaviour prevents exceptions from bubbling up for malformed files,
      but callers should be aware that lack of section info could be due to parse failure.
    """
    # Default feature values in case parsing fails or sections are missing
    features = {
        'text_section_entropy': 0.0,
        'data_section_entropy': 0.0,
        'rodata_section_entropy': 0.0,
        'bss_section_size': 0,
    }

    try:
        # Wrap raw data in a file-like object so pyelftools can parse it
        f = io.BytesIO(data)
        # Create ELFFile object which provides section iteration
        elf = ELFFile(f)

        # Iterate through sections to find named sections
        for section in elf.iter_sections():
            # Section name string (e.g., '.text', '.data', '.rodata', '.bss')
            name = section.name

            # If the section type is SHT_NOBITS (.bss), it has no stored bytes in file.
            # Use empty bytes for its data in that case.
            section_data = section.data() if section['sh_type'] != 'SHT_NOBITS' else b''

            # Match section names and compute/store the desired features.
            if name == '.text' and section_data:
                features['text_section_entropy'] = entropy(section_data)
            elif name == '.data' and section_data:
                features['data_section_entropy'] = entropy(section_data)
            elif name == '.rodata' and section_data:
                features['rodata_section_entropy'] = entropy(section_data)
            elif name == '.bss':
                # For .bss, record declared size (sh_size) because it occupies memory but not file bytes
                features['bss_section_size'] = section['sh_size']
    except Exception:
        # Silently ignore any exception (parse errors, truncated files); keep default features
        # Note: In a stricter implementation we might want to log the exception for debugging.
        pass

    return features


# ===========================
# Main feature extractor for Pipeline B.
# Combines all statistical feature groups into a single dictionary.
# ===========================
def extract_pipeline_b(data: bytes) -> dict:
    """
    Extract statistical features from ELF file.

    Parameters:
    - data (bytes): Raw bytes of the ELF file.

    Returns:
    - dict: A dictionary mapping feature name -> value. The keys correspond to the
      ordering returned by get_pipeline_b_feature_names().

    Explanation:
    - This function composes the other feature groups: global entropy, windowed entropy,
      byte histogram, n-gram features, header features, section entropies, and file size.
    - The returned dictionary may contain float and integer values. Missing features are
      not expected because all helper functions return consistent keys.
    """
    features = {}  # accumulator dictionary

    # Global entropy across the whole file
    features['global_entropy'] = entropy(data)

    # Windowed entropy statistics (min, max, mean, std)
    windowed = windowed_entropy(data)
    # Merge windowed stats into features
    features.update(windowed)

    # Coarse byte histogram features (16 bins + special frequencies)
    histogram = byte_histogram(data)
    features.update(histogram)

    # N-gram based features (for n=2,3,4 by default)
    ngrams = ngram_features(data)
    features.update(ngrams)

    # Header-specific features (first 1024 bytes and ELF header)
    header = header_features(data)
    features.update(header)

    # Section entropy features using ELF structural info where available
    section_ent = section_entropy_features(data)
    features.update(section_ent)

    # File size in bytes
    features['file_size'] = len(data)

    # Return the assembled features dictionary
    return features


# ===========================
# Fixed feature ordering used during ML training and inference.
# ===========================
def get_pipeline_b_feature_names():
    """
    Return list of feature names in order.

    Returns:
    - list[str]: Ordered list of keys corresponding to features returned by extract_pipeline_b()
                 This ordering is important when converting the feature dictionary to a vector
                 for ML models that expect fixed input ordering.
    """
    return [
        # Entropy
        'global_entropy', 'entropy_min', 'entropy_max', 'entropy_mean', 'entropy_std',
        # Byte histogram (16 bins)
        'byte_freq_bin_0', 'byte_freq_bin_1', 'byte_freq_bin_2', 'byte_freq_bin_3',
        'byte_freq_bin_4', 'byte_freq_bin_5', 'byte_freq_bin_6', 'byte_freq_bin_7',
        'byte_freq_bin_8', 'byte_freq_bin_9', 'byte_freq_bin_10', 'byte_freq_bin_11',
        'byte_freq_bin_12', 'byte_freq_bin_13', 'byte_freq_bin_14', 'byte_freq_bin_15',
        # Special bytes
        'null_byte_freq', 'printable_freq', 'high_byte_freq',
        # N-grams
        'ngram_2_unique', 'ngram_2_entropy', 'ngram_2_most_common_freq', 'ngram_2_diversity_50',
        'ngram_3_unique', 'ngram_3_entropy', 'ngram_3_most_common_freq', 'ngram_3_diversity_50',
        'ngram_4_unique', 'ngram_4_entropy', 'ngram_4_most_common_freq', 'ngram_4_diversity_50',
        # Header
        'header_entropy', 'header_null_freq', 'elf_header_entropy',
        # Section entropy
        'text_section_entropy', 'data_section_entropy', 'rodata_section_entropy', 'bss_section_size',
        # Size
        'file_size',
    ]


# ===========================
# Convert feature dictionary into ordered vector for ML prediction.
# ===========================
def extract_pipeline_b_vector(data: bytes) -> list:
    """
    Extract features as a vector (list) for ML.

    Parameters:
    - data (bytes): Raw bytes of the ELF file.

    Returns:
    - list: Ordered list of feature values matching get_pipeline_b_feature_names().

    Explanation:
    - This helper is used at inference time to produce a numeric vector in a
      consistent order expected by trained models. Missing keys are mapped to 0.
    """
    # Extract dictionary of features
    features = extract_pipeline_b(data)
    # Get the canonical order of feature names
    names = get_pipeline_b_feature_names()
    # Build and return the vector, using .get to default missing features to 0
    return [features.get(name, 0) for name in names]


# When the module is executed directly, run a simple CLI test
if __name__ == "__main__":
    # Import sys for command-line arguments (already used below)
    import sys

    # If a filepath argument is provided, open and analyze it
    if len(sys.argv) > 1:
        # Open file in binary mode so we can read raw bytes
        with open(sys.argv[1], 'rb') as f:
            data = f.read()  # read entire file into memory (careful with large files)

        # Extract features for the provided file
        features = extract_pipeline_b(data)

        # Print a short header reporting how many features were extracted
        print(f"Extracted {len(features)} features:")

        # Iterate through feature items in deterministic order (insertion order for Python dict)
        for k, v in features.items():
            # Format float values with 4 decimal places, leave integers as-is
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

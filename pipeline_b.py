
# ===========================
# Pipeline B — Statistical Feature Extraction
# This pipeline treats the ELF file as a raw byte sequence and extracts
# statistical patterns that may indicate packing, encryption, or obfuscation.
# Extracts statistical features from ELF file:
# - Entropy (global and windowed)
# - N-grams (byte patterns)
# - Byte distribution statistics
# ===========================
import math
from collections import Counter
import io
from elftools.elf.elffile import ELFFile

# ===========================
# Compute Shannon entropy of a byte sequence.
# High entropy may indicate compressed or encrypted regions.
# ===========================
def entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte sequence."""
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())



# ===========================
# Compute entropy over sliding windows across the file.
# This captures local entropy spikes instead of only global averages.
# ===========================
def windowed_entropy(data: bytes, window_size: int = 256, step: int = 128) -> dict:
    """Calculate entropy statistics over sliding windows."""
    if len(data) < window_size:
        return {
            'entropy_min': entropy(data),
            'entropy_max': entropy(data),
            'entropy_mean': entropy(data),
            'entropy_std': 0.0,
        }
    
    entropies = []
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]
        entropies.append(entropy(window))
    
    if not entropies:
        return {
            'entropy_min': 0.0,
            'entropy_max': 0.0,
            'entropy_mean': 0.0,
            'entropy_std': 0.0,
        }
    
    mean = sum(entropies) / len(entropies)
    variance = sum((e - mean) ** 2 for e in entropies) / len(entropies)
    
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
    """Calculate byte frequency distribution features."""
    if not data:
        return {f'byte_freq_{i}': 0.0 for i in range(16)}
    
    counts = Counter(data)
    total = len(data)
    
    # Divide 256 byte values into 16 bins (0-15, 16-31, etc.)
    features = {}
    for bin_idx in range(16):
        bin_count = sum(counts.get(b, 0) for b in range(bin_idx * 16, (bin_idx + 1) * 16))
        features[f'byte_freq_bin_{bin_idx}'] = bin_count / total
    
    # Special byte frequencies
    features['null_byte_freq'] = counts.get(0, 0) / total
    features['printable_freq'] = sum(counts.get(b, 0) for b in range(32, 127)) / total
    features['high_byte_freq'] = sum(counts.get(b, 0) for b in range(128, 256)) / total
    
    return features

# ===========================
# Extract N-gram statistics from byte stream (n = 2,3,4).
# These features capture short byte-pattern repetitions.
# ===========================
def ngram_features(data: bytes, n_values: list = [2, 3, 4]) -> dict:
    """Extract N-gram features from byte sequence."""
    features = {}
    
    for n in n_values:
        if len(data) < n:
            features[f'ngram_{n}_unique'] = 0
            features[f'ngram_{n}_entropy'] = 0.0
            features[f'ngram_{n}_most_common_freq'] = 0.0
            continue
        
        # Count N-grams
        ngrams = [tuple(data[i:i+n]) for i in range(len(data) - n + 1)]
        ngram_counts = Counter(ngrams)
        
        # Number of unique N-grams
        features[f'ngram_{n}_unique'] = len(ngram_counts)
        
        # N-gram entropy
        total = len(ngrams)
        ngram_ent = -sum((c/total) * math.log2(c/total) for c in ngram_counts.values())
        features[f'ngram_{n}_entropy'] = ngram_ent
        
        # Most common N-gram frequency
        most_common_count = ngram_counts.most_common(1)[0][1] if ngram_counts else 0
        features[f'ngram_{n}_most_common_freq'] = most_common_count / total if total > 0 else 0
        
        # Top N-gram diversity (how many N-grams needed to cover 50% of data)
        sorted_counts = sorted(ngram_counts.values(), reverse=True)
        cumsum = 0
        for i, count in enumerate(sorted_counts):
            cumsum += count
            if cumsum >= total * 0.5:
                features[f'ngram_{n}_diversity_50'] = (i + 1) / len(sorted_counts) if sorted_counts else 0
                break
        else:
            features[f'ngram_{n}_diversity_50'] = 1.0
    
    return features

# ===========================
# Analyze only the ELF header area (first ~1KB).
# Useful for detecting abnormal or obfuscated headers.
# ===========================
def header_features(data: bytes) -> dict:
    """Extract features from ELF header region."""
    # Focus on first 1024 bytes (header area)
    header_data = data[:1024] if len(data) > 1024 else data
    
    features = {}
    features['header_entropy'] = entropy(header_data)
    features['header_null_freq'] = header_data.count(0) / len(header_data) if header_data else 0
    
    # First 64 bytes (ELF header)
    elf_header = data[:64] if len(data) >= 64 else data
    features['elf_header_entropy'] = entropy(elf_header)
    
    return features

# ===========================
# Compute entropy of specific ELF sections (.text, .data, .rodata) and size of .bss.
# This partially reintroduces structural awareness into a statistical pipeline.
# ===========================
def section_entropy_features(data: bytes) -> dict:
    """Calculate entropy of each section type."""
    features = {
        'text_section_entropy': 0.0,
        'data_section_entropy': 0.0,
        'rodata_section_entropy': 0.0,
        'bss_section_size': 0,
    }
    
    try:
        f = io.BytesIO(data)
        elf = ELFFile(f)
        
        for section in elf.iter_sections():
            name = section.name
            section_data = section.data() if section['sh_type'] != 'SHT_NOBITS' else b''
            
            if name == '.text' and section_data:
                features['text_section_entropy'] = entropy(section_data)
            elif name == '.data' and section_data:
                features['data_section_entropy'] = entropy(section_data)
            elif name == '.rodata' and section_data:
                features['rodata_section_entropy'] = entropy(section_data)
            elif name == '.bss':
                features['bss_section_size'] = section['sh_size']
    except:
        pass
    
    return features


# ===========================
# Main feature extractor for Pipeline B.
# Combines all statistical feature groups into a single dictionary.
# ===========================
def extract_pipeline_b(data: bytes) -> dict:
    """
    Extract statistical features from ELF file.
    Returns a dictionary of features.
    """
    features = {}
    
    # Global entropy
    features['global_entropy'] = entropy(data)
    
    # Windowed entropy
    windowed = windowed_entropy(data)
    features.update(windowed)
    
    # Byte histogram
    histogram = byte_histogram(data)
    features.update(histogram)
    
    # N-gram features
    ngrams = ngram_features(data)
    features.update(ngrams)
    
    # Header-specific features
    header = header_features(data)
    features.update(header)
    
    # Section entropy
    section_ent = section_entropy_features(data)
    features.update(section_ent)
    
    # File size
    features['file_size'] = len(data)
    
    return features


# ===========================
# Fixed feature ordering used during ML training and inference.
# ===========================
def get_pipeline_b_feature_names():
    """Return list of feature names in order."""
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
    """Extract features as a vector (list) for ML."""
    features = extract_pipeline_b(data)
    names = get_pipeline_b_feature_names()
    return [features.get(name, 0) for name in names]


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            data = f.read()
        features = extract_pipeline_b(data)
        print(f"Extracted {len(features)} features:")
        for k, v in features.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

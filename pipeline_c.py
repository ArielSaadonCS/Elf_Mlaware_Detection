"""
Pipeline C: Hybrid Approach (Our Improvement)

Combines the best of Pipeline A and B with additional improvements:
1. Feature selection from both pipelines
2. New derived features
3. Anomaly detection features
4. Ensemble with stacking

This file contains detailed, line-by-line comments and expanded
docstrings so you can understand exactly what each statement does
and how data flows through the feature extraction pipeline.
"""
# ===========================
# Pipeline C — Hybrid Feature Extraction (Our Improvement)
# This pipeline combines selected structural features (Pipeline A),
# selected statistical features (Pipeline B), and new derived security
# and anomaly features designed to better capture real malware behavior.
# ===========================

# Standard library imports
import math                       # For math.sqrt and other math operations
from collections import Counter   # Imported earlier for potential aggregation use (unused here)
import io                         # For BytesIO to treat bytes as a file-like object

# Third-party imports from pyelftools
from elftools.elf.elffile import ELFFile             # Main class to parse ELF files
from elftools.elf.sections import SymbolTableSection # Type check for symbol tables
from elftools.elf.dynamic import DynamicSection      # Imported for potential dynamic section checks (unused here)

# ===========================
# Reuse existing feature extractors from Pipelines A and B.
# This avoids duplicating logic and keeps consistency with training.
# ===========================
from pipeline_a import extract_pipeline_a, get_pipeline_a_feature_names
# Import feature extraction and name utility from pipeline A

from pipeline_b import extract_pipeline_b, get_pipeline_b_feature_names, entropy
# Import feature extraction, name utility, and entropy function from pipeline B
# `entropy` is reused here to compute section entropies

# ===========================
# Extract novel derived features that mix structure, statistics, and
# security-oriented indicators. These represent the main contribution
# of the hybrid pipeline.
# ===========================
def extract_derived_features(data: bytes) -> dict:
    """
    Extract new derived features that combine structural and statistical info.

    Parameters
    - data (bytes): Raw bytes of an ELF file.

    Returns
    - dict: Mapping of derived feature names to values.

    Purpose
    - Parse the ELF binary from bytes and compute ratios, entropy-based
      anomaly features, security flags (NX/PIE/RELRO), suspicious section
      and import indicators, and string-based indicators such as URLs or IPs.
    """
    # Initialize feature dictionary to accumulate results
    features = {}
    
    try:
        # Create a file-like object from bytes to feed into ELFFile
        f = io.BytesIO(data)
        # Parse the ELF structure from the in-memory bytes
        elf = ELFFile(f)
        
        # ============ Ratio Features ============
        # Capture relationships between code, data, headers, and total file size.
        # These ratios can indicate packing, padding, or abnormal layout.
        file_size = len(data)                      # Total size of the file in bytes
        
        # Initialize section size counters to 0
        text_size = 0                              # Size of .text (code)
        data_size = 0                              # Accumulated size of data-like sections
        
        # Iterate over sections to find sizes for .text and data-like sections
        for section in elf.iter_sections():
            name = section.name                     # Section name, e.g., '.text'
            if name == '.text':
                # Get section size from section header field sh_size
                text_size = section['sh_size']
            elif name in ['.data', '.rodata']:
                # Add sizes for .data and .rodata to approximate data footprint
                data_size += section['sh_size']
        
        # Compute code-to-data ratio, guarding against division by zero
        features['code_to_data_ratio'] = text_size / data_size if data_size > 0 else 0
        # Compute code-to-file ratio, guarding against zero file size
        features['code_to_file_ratio'] = text_size / file_size if file_size > 0 else 0
        
        # Header to file ratio: ELF header size plus program header table size
        # elf.header['e_ehsize'] is the ELF header size
        # elf.header['e_phnum'] * elf.header['e_phentsize'] approximates program header table total size
        header_size = elf.header['e_ehsize'] + (elf.header['e_phnum'] * elf.header['e_phentsize'])
        features['header_to_file_ratio'] = header_size / file_size if file_size > 0 else 0

        # ===========================
        # ENTROPY ANOMALY FEATURES
        # Detect abnormal entropy distribution across sections.
        # High entropy sections often indicate encryption or packing.
        # ===========================
        section_entropies = []                     # Will hold entropy for each relevant section
        # Iterate sections to compute entropy for non-SHT_NOBITS sections (they have data)
        for section in elf.iter_sections():
            # Skip sections that don't actually occupy file space (SHT_NOBITS) or have zero size
            if section['sh_type'] != 'SHT_NOBITS' and section['sh_size'] > 0:
                try:
                    # Read section bytes
                    section_data = section.data()
                    # If we successfully retrieved section bytes, compute entropy
                    if section_data:
                        section_entropies.append(entropy(section_data))
                except:
                    # If reading section data fails for some reason, skip it silently
                    # (we prefer robust feature extraction over failing)
                    pass
        
        # If we collected any entropies, compute aggregate statistics
        if section_entropies:
            # Mean entropy across sections
            features['section_entropy_mean'] = sum(section_entropies) / len(section_entropies)
            # Maximum entropy found among sections
            features['section_entropy_max'] = max(section_entropies)
            # Standard deviation of section entropies (population std dev)
            features['section_entropy_std'] = math.sqrt(
                sum((e - features['section_entropy_mean'])**2 for e in section_entropies) / len(section_entropies)
            )
            # Count of sections with entropy > 7.0 (heuristic threshold for packed/encrypted)
            features['high_entropy_section_count'] = sum(1 for e in section_entropies if e > 7.0)
        else:
            # If no entropies were found, set defaults to zero
            features['section_entropy_mean'] = 0
            features['section_entropy_max'] = 0
            features['section_entropy_std'] = 0
            features['high_entropy_section_count'] = 0
        
        # ============ Security Features ============
        # Check for security-related sections/flags: stack canary, PIE, RELRO, NX
        # Initialize flags to zero (false)
        has_stack_canary = 0  # Placeholder; detection not implemented in this function
        has_pie = 0           # Position Independent Executable flag presence
        has_relro = 0         # RELRO presence flag
        has_nx = 0            # Non-executable stack (NX) presence flag
        
        # Iterate program headers (segments) to detect PT_GNU_STACK and PT_GNU_RELRO
        for segment in elf.iter_segments():
            p_type = segment['p_type']  # e.g., 'PT_LOAD', 'PT_GNU_STACK'
            if p_type == 'PT_GNU_STACK':
                # Check NX: if execute bit (0x1) is not set, stack is non-executable
                # segment['p_flags'] is an int flags field; & 0x1 checks the execute bit
                if not (segment['p_flags'] & 0x1):  # No execute flag means NX is enabled
                    has_nx = 1
            elif p_type == 'PT_GNU_RELRO':
                # If RELRO segment present, set has_relro flag
                has_relro = 1
        
        # PIE detection: ET_DYN binaries are often used for PIE, but confirm by checking for an INTERP segment
        if elf.header['e_type'] == 'ET_DYN':
            # If the ELF has program interpreter (PT_INTERP), treat as PIE candidate and set has_pie
            for segment in elf.iter_segments():
                if segment['p_type'] == 'PT_INTERP':
                    has_pie = 1
                    break  # No need to continue once PIE is detected
        
        # Store security indicator features into dictionary
        features['has_nx'] = has_nx
        features['has_pie'] = has_pie
        features['has_relro'] = has_relro
        
        # ============ Suspicious Pattern Features ============
        # Count suspicious section names often used by packers or obfuscators
        suspicious_names = ['.packed', '.upx', '.aspack', '.petite', '.mpress']  # common packer markers
        suspicious_count = 0
        for section in elf.iter_sections():
            # Convert section name to lowercase and check for substrings indicating packing
            if any(s in section.name.lower() for s in suspicious_names):
                suspicious_count += 1
        features['suspicious_section_count'] = suspicious_count
        
        # Count sections with empty or null names (sometimes used to hide sections)
        empty_name_count = 0
        for section in elf.iter_sections():
            # section.name could be '' or contain only whitespace; count those
            if not section.name or section.name.strip() == '':
                empty_name_count += 1
        features['empty_section_name_count'] = empty_name_count
        
        # ============ Import Analysis ============
        # Suspicious imports often used by malware; look for their occurrence in symbol tables
        suspicious_imports = ['execve', 'fork', 'socket', 'connect', 'bind', 
                            'chmod', 'chown', 'unlink', 'ptrace', 'mprotect',
                            'dlopen', 'system', 'popen']
        suspicious_import_count = 0
        
        # Iterate over sections and find symbol tables to inspect imported symbol names
        for section in elf.iter_sections():
            # Only process sections that are instances of SymbolTableSection
            if isinstance(section, SymbolTableSection):
                # Iterate over symbols in the symbol table
                for symbol in section.iter_symbols():
                    # Symbol names can be None or empty; lower them for case-insensitive matching
                    sym_name = symbol.name.lower()
                    # If any suspicious substring appears in the symbol name, increment the counter
                    if any(s in sym_name for s in suspicious_imports):
                        suspicious_import_count += 1
        
        # Save suspicious import count into features dictionary
        features['suspicious_import_count'] = suspicious_import_count
        
        # ===========================
        # STRING-BASED INDICATORS
        # Count embedded strings and detect URLs / IP addresses.
        # ===========================
        strings = extract_strings(data)  # Extract printable ASCII strings from the raw bytes
        features['num_strings'] = len(strings)  # Total number of strings found
        # Average string length; avoid division by zero if no strings found
        features['avg_string_length'] = sum(len(s) for s in strings) / len(strings) if strings else 0
        
        # Count strings containing common URL patterns (http, www)
        url_count = sum(1 for s in strings if 'http' in s.lower() or 'www.' in s.lower())
        # Count strings that look like IPv4 addresses using a helper function
        ip_count = sum(1 for s in strings if is_ip_pattern(s))
        features['url_string_count'] = url_count
        features['ip_string_count'] = ip_count
        
    except Exception as e:
        # If any error occurs during parsing or extraction, return default zeroed features
        # This keeps feature extraction robust for malformed or unsupported files
        features = {
            'code_to_data_ratio': 0,
            'code_to_file_ratio': 0,
            'header_to_file_ratio': 0,
            'section_entropy_mean': 0,
            'section_entropy_max': 0,
            'section_entropy_std': 0,
            'high_entropy_section_count': 0,
            'has_nx': 0,
            'has_pie': 0,
            'has_relro': 0,
            'suspicious_section_count': 0,
            'empty_section_name_count': 0,
            'suspicious_import_count': 0,
            'num_strings': 0,
            'avg_string_length': 0,
            'url_string_count': 0,
            'ip_string_count': 0,
        }
    
    # Return the dictionary of derived features
    return features

# ===========================
# Extract printable ASCII strings from binary data.
# ===========================
def extract_strings(data: bytes, min_length: int = 4) -> list:
    """
    Extract ASCII strings from binary data.

    Parameters
    - data (bytes): Raw bytes to search for printable ASCII sequences.
    - min_length (int): Minimum length of a sequence to be considered a string (default 4).

    Returns
    - list: A list of extracted strings (type: str).

    Purpose
    - Finds contiguous runs of printable ASCII characters (bytes in range 32..126)
      and returns them as Python strings if they meet the minimum length.
    """
    strings = []     # Accumulator for found strings
    current = []     # Characters in the current potential string run
    
    # Iterate over each byte in the input data
    for byte in data:
        # Check for printable ASCII range (32..126 inclusive)
        if 32 <= byte < 127:  # If byte corresponds to a printable ASCII character
            current.append(chr(byte))  # Append the character to the current run
        else:
            # If we encounter a non-printable byte, evaluate the current run
            if len(current) >= min_length:
                # Join characters into a string and append to results
                strings.append(''.join(current))
            # Reset the current run for the next potential string
            current = []
    
    # After the loop, we may have a trailing run to evaluate
    if len(current) >= min_length:
        strings.append(''.join(current))
    
    # Return the list of extracted strings
    return strings

# ===========================
# Check whether a string resembles an IPv4 address.
# ===========================
def is_ip_pattern(s: str) -> bool:
    """
    Check if a string looks like an IPv4 address.

    Parameters
    - s (str): String to evaluate.

    Returns
    - bool: True if s appears to be an IPv4 address in dotted-decimal notation.

    Purpose
    - Provide a simple heuristic to detect strings that are likely IPv4 addresses.
      Only accepts 4 dot-separated numeric parts and each part must be between 0 and 255.
    """
    parts = s.split('.')  # Split on dot
    # If not exactly 4 parts, can't be a valid IPv4 dotted-decimal address
    if len(parts) != 4:
        return False
    try:
        # Convert each part to int and check bounds 0..255 inclusive
        return all(0 <= int(p) <= 255 for p in parts)
    except:
        # If conversion to int fails for any part, it's not an IP address
        return False

# ===========================
# Combine selected features from Pipeline A, Pipeline B,
# and all derived features into a single hybrid feature set.
# ===========================
def extract_pipeline_c(data: bytes) -> dict:
    """
    Extract combined features from Pipeline C.

    Parameters
    - data (bytes): Raw bytes of the ELF binary to analyze.

    Returns
    - dict: Mapping of feature names (prefixed with a_/b_/c_) to feature values.

    Purpose
    - Call Pipeline A and Pipeline B feature extractors, compute derived features,
      then select and prefix features from each source to build a unified feature
      dictionary consumed by downstream ML models.
    """
    # Get features from Pipeline A (structural features)
    features_a = extract_pipeline_a(data)
    # Get features from Pipeline B (statistical features)
    features_b = extract_pipeline_b(data)
    # Compute derived features implemented above
    derived = extract_derived_features(data)
    
    # Combined features dictionary to be returned
    features = {}
    
    # Selected structural features from Pipeline A
    structural_keys = [
        'e_type', 'e_machine', 'num_sections', 'num_segments',
        'sections_executable', 'sections_writable', 'segments_executable',
        'num_symbols', 'num_functions', 'num_undefined_symbols',
        'num_libraries_needed', 'file_size'
    ]
    # Prefix structural keys with 'a_' to avoid collisions and indicate source
    for key in structural_keys:
        features[f'a_{key}'] = features_a.get(key, 0)  # Use .get with default 0 if missing
    
    # Selected statistical features from Pipeline B
    statistical_keys = [
        'global_entropy', 'entropy_mean', 'entropy_std',
        'null_byte_freq', 'printable_freq', 'high_byte_freq',
        'ngram_2_entropy', 'ngram_3_entropy',
        'header_entropy', 'text_section_entropy'
    ]
    # Prefix statistical keys with 'b_'
    for key in statistical_keys:
        features[f'b_{key}'] = features_b.get(key, 0)
    
    # Add all derived features with 'c_' prefix
    for key, value in derived.items():
        features[f'c_{key}'] = value
    
    # Return the full combined feature mapping
    return features

# ===========================
# Fixed ordering of Pipeline C features for ML compatibility.
# ===========================
def get_pipeline_c_feature_names():
    """
    Return list of feature names in a fixed, deterministic order.

    Purpose
    - Many ML models require a consistent feature ordering. This function
      enumerates the exact ordering for pipeline C so feature vectors
      can be constructed reliably with the same ordering used during training.
    """
    return [
        # From Pipeline A (structural)
        'a_e_type', 'a_e_machine', 'a_num_sections', 'a_num_segments',
        'a_sections_executable', 'a_sections_writable', 'a_segments_executable',
        'a_num_symbols', 'a_num_functions', 'a_num_undefined_symbols',
        'a_num_libraries_needed', 'a_file_size',
        # From Pipeline B (statistical)
        'b_global_entropy', 'b_entropy_mean', 'b_entropy_std',
        'b_null_byte_freq', 'b_printable_freq', 'b_high_byte_freq',
        'b_ngram_2_entropy', 'b_ngram_3_entropy',
        'b_header_entropy', 'b_text_section_entropy',
        # Derived features (our contribution)
        'c_code_to_data_ratio', 'c_code_to_file_ratio', 'c_header_to_file_ratio',
        'c_section_entropy_mean', 'c_section_entropy_max', 'c_section_entropy_std',
        'c_high_entropy_section_count',
        'c_has_nx', 'c_has_pie', 'c_has_relro',
        'c_suspicious_section_count', 'c_empty_section_name_count',
        'c_suspicious_import_count',
        'c_num_strings', 'c_avg_string_length',
        'c_url_string_count', 'c_ip_string_count',
    ]


# ===========================
# Convert Pipeline C features into ML input vector.
# ===========================
def extract_pipeline_c_vector(data: bytes) -> list:
    """
    Extract features as a vector (list) for ML.

    Parameters
    - data (bytes): Raw bytes of ELF file to process.

    Returns
    - list: Feature values in the order specified by get_pipeline_c_feature_names().

    Purpose
    - Produce a numerical list matching the model's expected input shape.
    """
    # Extract full feature dictionary
    features = extract_pipeline_c(data)
    # Retrieve ordered names to ensure deterministic vector ordering
    names = get_pipeline_c_feature_names()
    # Build a list of values matching that order; default to 0 if a name is missing
    return [features.get(name, 0) for name in names]


# If this file is executed directly (not imported), run a simple test
if __name__ == "__main__":
    # Import sys to read the filename argument from the command line
    import sys
    # Only proceed if an argument (path to a file) was provided
    if len(sys.argv) > 1:
        # Open the specified file in binary mode and read its contents
        with open(sys.argv[1], 'rb') as f:
            data = f.read()
        # Run pipeline C extraction
        features = extract_pipeline_c(data)
        # Print a summary listing number of features and grouped sections
        print(f"Pipeline C: Extracted {len(features)} features:")
        print("\n=== Structural (from A) ===")
        # Print structural features (those prefixed with 'a_')
        for k, v in features.items():
            if k.startswith('a_'):
                print(f"  {k}: {v}")
        print("\n=== Statistical (from B) ===")
        # Print statistical features (those prefixed with 'b_'); format floats nicely
        for k, v in features.items():
            if k.startswith('b_'):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("\n=== Derived (our contribution) ===")
        # Print derived features (those prefixed with 'c_'); format floats nicely
        for k, v in features.items():
            if k.startswith('c_'):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

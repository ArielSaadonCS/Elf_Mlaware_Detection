"""
Pipeline C: Hybrid Approach (Our Improvement)

Combines the best of Pipeline A and B with additional improvements:
1. Feature selection from both pipelines
2. New derived features
3. Anomaly detection features
4. Ensemble with stacking
"""
# ===========================
# Pipeline C — Hybrid Feature Extraction (Our Improvement)
# This pipeline combines selected structural features (Pipeline A),
# selected statistical features (Pipeline B), and new derived security
# and anomaly features designed to better capture real malware behavior.
# ===========================
import math
from collections import Counter
import io
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from elftools.elf.dynamic import DynamicSection


# ===========================
# Reuse existing feature extractors from Pipelines A and B.
# This avoids duplicating logic and keeps consistency with training.
# ===========================
from pipeline_a import extract_pipeline_a, get_pipeline_a_feature_names
from pipeline_b import extract_pipeline_b, get_pipeline_b_feature_names, entropy



# ===========================
# Extract novel derived features that mix structure, statistics, and
# security-oriented indicators. These represent the main contribution
# of the hybrid pipeline.
# ===========================
def extract_derived_features(data: bytes) -> dict:
    """
    Extract new derived features that combine structural and statistical info.
    These are our novel contributions.
    """
    features = {}
    
    try:
        f = io.BytesIO(data)
        elf = ELFFile(f)
        
        # ============ Ratio Features ============
        # Capture relationships between code, data, headers, and total file size.
        file_size = len(data)
        
        text_size = 0
        data_size = 0
        for section in elf.iter_sections():
            name = section.name
            if name == '.text':
                text_size = section['sh_size']
            elif name in ['.data', '.rodata']:
                data_size += section['sh_size']
        
        features['code_to_data_ratio'] = text_size / data_size if data_size > 0 else 0
        features['code_to_file_ratio'] = text_size / file_size if file_size > 0 else 0
        
        # Header to file ratio
        header_size = elf.header['e_ehsize'] + (elf.header['e_phnum'] * elf.header['e_phentsize'])
        features['header_to_file_ratio'] = header_size / file_size if file_size > 0 else 0

        # ===========================
        # ENTROPY ANOMALY FEATURES
        # Detect abnormal entropy distribution across sections.
        # ===========================
        section_entropies = []
        for section in elf.iter_sections():
            if section['sh_type'] != 'SHT_NOBITS' and section['sh_size'] > 0:
                try:
                    section_data = section.data()
                    if section_data:
                        section_entropies.append(entropy(section_data))
                except:
                    pass
        
        if section_entropies:
            features['section_entropy_mean'] = sum(section_entropies) / len(section_entropies)
            features['section_entropy_max'] = max(section_entropies)
            features['section_entropy_std'] = math.sqrt(
                sum((e - features['section_entropy_mean'])**2 for e in section_entropies) / len(section_entropies)
            )
            # High entropy sections (>7.0) often indicate encryption/packing
            features['high_entropy_section_count'] = sum(1 for e in section_entropies if e > 7.0)
        else:
            features['section_entropy_mean'] = 0
            features['section_entropy_max'] = 0
            features['section_entropy_std'] = 0
            features['high_entropy_section_count'] = 0
        
        # ============ Security Features ============
        # Check for security-related sections/flags
        has_stack_canary = 0
        has_pie = 0
        has_relro = 0
        has_nx = 0
        
        for segment in elf.iter_segments():
            p_type = segment['p_type']
            if p_type == 'PT_GNU_STACK':
                # Non-executable stack (NX)
                if not (segment['p_flags'] & 0x1):  # No execute flag
                    has_nx = 1
            elif p_type == 'PT_GNU_RELRO':
                has_relro = 1
        
        # PIE detection (DYN type with INTERP)
        if elf.header['e_type'] == 'ET_DYN':
            for segment in elf.iter_segments():
                if segment['p_type'] == 'PT_INTERP':
                    has_pie = 1
                    break
        
        features['has_nx'] = has_nx
        features['has_pie'] = has_pie
        features['has_relro'] = has_relro
        
        # ============ Suspicious Pattern Features ============
        # Count suspicious section names
        suspicious_names = ['.packed', '.upx', '.aspack', '.petite', '.mpress']
        suspicious_count = 0
        for section in elf.iter_sections():
            if any(s in section.name.lower() for s in suspicious_names):
                suspicious_count += 1
        features['suspicious_section_count'] = suspicious_count
        
        # Empty/null section names
        empty_name_count = 0
        for section in elf.iter_sections():
            if not section.name or section.name.strip() == '':
                empty_name_count += 1
        features['empty_section_name_count'] = empty_name_count
        
        # ============ Import Analysis ============
        # Suspicious imports often used by malware
        suspicious_imports = ['execve', 'fork', 'socket', 'connect', 'bind', 
                            'chmod', 'chown', 'unlink', 'ptrace', 'mprotect',
                            'dlopen', 'system', 'popen']
        suspicious_import_count = 0
        
        for section in elf.iter_sections():
            if isinstance(section, SymbolTableSection):
                for symbol in section.iter_symbols():
                    sym_name = symbol.name.lower()
                    if any(s in sym_name for s in suspicious_imports):
                        suspicious_import_count += 1
        
        features['suspicious_import_count'] = suspicious_import_count
        
        # ===========================
        # STRING-BASED INDICATORS
        # Count embedded strings and detect URLs / IP addresses.
        # ===========================
        strings = extract_strings(data)
        features['num_strings'] = len(strings)
        features['avg_string_length'] = sum(len(s) for s in strings) / len(strings) if strings else 0
        
        # URL/IP patterns (common in malware)
        url_count = sum(1 for s in strings if 'http' in s.lower() or 'www.' in s.lower())
        ip_count = sum(1 for s in strings if is_ip_pattern(s))
        features['url_string_count'] = url_count
        features['ip_string_count'] = ip_count
        
    except Exception as e:
        # Return default values on error
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
    
    return features

# ===========================
# Extract printable ASCII strings from binary data.
# ===========================
def extract_strings(data: bytes, min_length: int = 4) -> list:
    """Extract ASCII strings from binary data."""
    strings = []
    current = []
    
    for byte in data:
        if 32 <= byte < 127:  # Printable ASCII
            current.append(chr(byte))
        else:
            if len(current) >= min_length:
                strings.append(''.join(current))
            current = []
    
    if len(current) >= min_length:
        strings.append(''.join(current))
    
    return strings

# ===========================
# Check whether a string resembles an IPv4 address.
# ===========================
def is_ip_pattern(s: str) -> bool:
    """Check if string looks like an IP address."""
    parts = s.split('.')
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 for p in parts)
    except:
        return False

# ===========================
# Combine selected features from Pipeline A, Pipeline B,
# and all derived features into a single hybrid feature set.
# ===========================
def extract_pipeline_c(data: bytes) -> dict:
    """
    Extract combined features from Pipeline C.
    Combines selected features from A and B with derived features.
    """
    # Get features from both pipelines
    features_a = extract_pipeline_a(data)
    features_b = extract_pipeline_b(data)
    derived = extract_derived_features(data)
    
    # Combine all features
    features = {}
    
    # Selected structural features from Pipeline A
    structural_keys = [
        'e_type', 'e_machine', 'num_sections', 'num_segments',
        'sections_executable', 'sections_writable', 'segments_executable',
        'num_symbols', 'num_functions', 'num_undefined_symbols',
        'num_libraries_needed', 'file_size'
    ]
    for key in structural_keys:
        features[f'a_{key}'] = features_a.get(key, 0)
    
    # Selected statistical features from Pipeline B
    statistical_keys = [
        'global_entropy', 'entropy_mean', 'entropy_std',
        'null_byte_freq', 'printable_freq', 'high_byte_freq',
        'ngram_2_entropy', 'ngram_3_entropy',
        'header_entropy', 'text_section_entropy'
    ]
    for key in statistical_keys:
        features[f'b_{key}'] = features_b.get(key, 0)
    
    # All derived features
    for key, value in derived.items():
        features[f'c_{key}'] = value
    
    return features

# ===========================
# Fixed ordering of Pipeline C features for ML compatibility.
# ===========================
def get_pipeline_c_feature_names():
    """Return list of feature names in order."""
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
    """Extract features as a vector (list) for ML."""
    features = extract_pipeline_c(data)
    names = get_pipeline_c_feature_names()
    return [features.get(name, 0) for name in names]


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            data = f.read()
        features = extract_pipeline_c(data)
        print(f"Pipeline C: Extracted {len(features)} features:")
        print("\n=== Structural (from A) ===")
        for k, v in features.items():
            if k.startswith('a_'):
                print(f"  {k}: {v}")
        print("\n=== Statistical (from B) ===")
        for k, v in features.items():
            if k.startswith('b_'):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("\n=== Derived (our contribution) ===")
        for k, v in features.items():
            if k.startswith('c_'):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

"""
ELF Malware Detection Worker
Uses all three pipelines for analysis and comparison.
"""


import os
import io
import base64
import json
import math
import sys
from collections import Counter
from redis import Redis
import joblib
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from elftools.elf.dynamic import DynamicSection

# ------------------------------------------------------------
# Configuration:
# Read Redis connection details from env (with defaults for Docker Compose),
# and define the Redis list key that acts as the job queue.
# ------------------------------------------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_KEY = "elf_queue"

# ------------------------------------------------------------
# Redis client:
# Create a Redis client. decode_responses=True means Redis values are returned as strings,
# which is convenient since files are stored as base64 strings.
# ------------------------------------------------------------

redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================================================
# Load Models
# ============================================================

# ------------------------------------------------------------
# Model loading:
# Load 6 pre-trained models from disk (RF + XGBoost for pipelines A/B/C).
# If anything fails (missing file, incompatible pickle, etc.), exit immediately
# because the worker cannot function without the models.
# ------------------------------------------------------------

print("Loading models...")
models = {}
try:
    models['pipeline_a_rf'] = joblib.load("/app/models/pipeline_a_rf.pkl")
    models['pipeline_a_xgb'] = joblib.load("/app/models/pipeline_a_xgb.pkl")
    models['pipeline_b_rf'] = joblib.load("/app/models/pipeline_b_rf.pkl")
    models['pipeline_b_xgb'] = joblib.load("/app/models/pipeline_b_xgb.pkl")
    models['pipeline_c_rf'] = joblib.load("/app/models/pipeline_c_rf.pkl")
    models['pipeline_c_xgb'] = joblib.load("/app/models/pipeline_c_xgb.pkl")
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# ============================================================
# Mappings
# ============================================================

# ------------------------------------------------------------
# Categorical-to-numeric mappings:
# we map known values to stable integers.
# ------------------------------------------------------------

E_TYPE_MAP = {
    'ET_NONE': 0, 'ET_REL': 1, 'ET_EXEC': 2, 'ET_DYN': 3, 'ET_CORE': 4,
}

E_MACHINE_MAP = {
    'EM_NONE': 0, 'EM_386': 3, 'EM_ARM': 40, 'EM_X86_64': 62,
    'EM_AARCH64': 183, 'EM_MIPS': 8, 'EM_PPC': 20, 'EM_PPC64': 21,
}

# ------------------------------------------------------------
# get_value():
# Normalizes ELF values into integers:
# - If it's a string enum: map it; if unknown, hash it into a bounded range.
# - If it's an object with .value: use that.
# - Otherwise: cast to int (or 0 if missing).
# -----------------------------------------------------------

def get_value(raw, mapping):
    if isinstance(raw, str):
        return mapping.get(raw, hash(raw) % 1000)
    elif hasattr(raw, 'value'):
        return raw.value
    else:
        return int(raw) if raw else 0

# ------------------------------------------------------------
# entropy():
# Computes Shannon entropy over bytes. 
# Higher entropy often indicates compression/encryption/packing.
# Used by pipeline B and parts of pipeline C.
# ------------------------------------------------------------

def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

# ============================================================
# Pipeline A: Structural Features (53 features)
# ============================================================

# ------------------------------------------------------------
# extract_pipeline_a():
# Parses the ELF structure and extracts a fixed-size vector of *structural* features:
# - Header fields (e_type, e_machine, offsets, counts, etc.)
# - Section stats (counts by type, flags, size min/max/mean/total)
# - Segment stats (counts by type, flags)
# - Symbol table stats (how many funcs/undefined/global/etc.)
# - Dynamic section stats (DT_NEEDED, RPATH/RUNPATH/SONAME)
# Finally returns features in a specific key order to match training-time columns.
# ------------------------------------------------------------

def extract_pipeline_a(data: bytes) -> list:
    f = io.BytesIO(data)
    elf = ELFFile(f)
    header = elf.header
    features = {}
    
    # ELF Header
    # Header features:
    # Directly take key fields from ELF header; map enums to integers.
    features['e_type'] = get_value(header['e_type'], E_TYPE_MAP)
    features['e_machine'] = get_value(header['e_machine'], E_MACHINE_MAP)
    e_version = header['e_version']
    features['e_version'] = 1 if isinstance(e_version, str) and 'CURRENT' in e_version else (int(e_version) if e_version else 0)
    features['e_entry'] = header['e_entry']
    features['e_phoff'] = header['e_phoff']
    features['e_shoff'] = header['e_shoff']
    features['e_flags'] = header['e_flags']
    features['e_ehsize'] = header['e_ehsize']
    features['e_phentsize'] = header['e_phentsize']
    features['e_phnum'] = header['e_phnum']
    features['e_shentsize'] = header['e_shentsize']
    features['e_shnum'] = header['e_shnum']
    features['e_shstrndx'] = header['e_shstrndx']
    
    # Sections
    features['num_sections'] = elf.num_sections()
    section_types = {}
    section_sizes = []
    section_flags_exec = section_flags_write = section_flags_alloc = 0
    
    for section in elf.iter_sections():
        sh_type = section['sh_type']
        type_name = sh_type if isinstance(sh_type, str) else f"TYPE_{sh_type}"
        section_types[type_name] = section_types.get(type_name, 0) + 1
        section_sizes.append(section['sh_size'])
        flags = section['sh_flags']
        if flags & 0x1: section_flags_write += 1
        if flags & 0x2: section_flags_alloc += 1
        if flags & 0x4: section_flags_exec += 1
    
    features['sections_executable'] = section_flags_exec
    features['sections_writable'] = section_flags_write
    features['sections_allocatable'] = section_flags_alloc
    features['sections_progbits'] = section_types.get('SHT_PROGBITS', 0)
    features['sections_nobits'] = section_types.get('SHT_NOBITS', 0)
    features['sections_symtab'] = section_types.get('SHT_SYMTAB', 0)
    features['sections_dynsym'] = section_types.get('SHT_DYNSYM', 0)
    features['sections_strtab'] = section_types.get('SHT_STRTAB', 0)
    features['sections_rela'] = section_types.get('SHT_RELA', 0)
    features['sections_rel'] = section_types.get('SHT_REL', 0)
    features['sections_dynamic'] = section_types.get('SHT_DYNAMIC', 0)
    features['sections_note'] = section_types.get('SHT_NOTE', 0)
    
    if section_sizes:
        features['section_size_min'] = min(section_sizes)
        features['section_size_max'] = max(section_sizes)
        features['section_size_mean'] = sum(section_sizes) / len(section_sizes)
        features['section_size_total'] = sum(section_sizes)
    else:
        features['section_size_min'] = features['section_size_max'] = features['section_size_mean'] = features['section_size_total'] = 0
    
    # Segments
    features['num_segments'] = elf.num_segments()
    segment_types = {}
    segment_flags_exec = segment_flags_write = segment_flags_read = 0
    
    for segment in elf.iter_segments():
        p_type = segment['p_type']
        type_name = p_type if isinstance(p_type, str) else f"TYPE_{p_type}"
        segment_types[type_name] = segment_types.get(type_name, 0) + 1
        flags = segment['p_flags']
        if flags & 0x1: segment_flags_exec += 1
        if flags & 0x2: segment_flags_write += 1
        if flags & 0x4: segment_flags_read += 1
    
    features['segments_executable'] = segment_flags_exec
    features['segments_writable'] = segment_flags_write
    features['segments_readable'] = segment_flags_read
    features['segments_load'] = segment_types.get('PT_LOAD', 0)
    features['segments_dynamic'] = segment_types.get('PT_DYNAMIC', 0)
    features['segments_interp'] = segment_types.get('PT_INTERP', 0)
    features['segments_note'] = segment_types.get('PT_NOTE', 0)
    features['segments_gnu_stack'] = segment_types.get('PT_GNU_STACK', 0)
    features['segments_gnu_relro'] = segment_types.get('PT_GNU_RELRO', 0)
    
    # Symbols
    # Iterate over symbol tables and count types/bindings/undefined symbols.
    # These capture how much the binary exposes (stripped vs not, dynamic vs static patterns).
    num_symbols = num_functions = num_objects = num_global = num_local = num_weak = num_undefined = 0
    for section in elf.iter_sections():
        if isinstance(section, SymbolTableSection):
            for symbol in section.iter_symbols():
                num_symbols += 1
                sym_type = symbol['st_info']['type']
                if sym_type == 'STT_FUNC': num_functions += 1
                elif sym_type == 'STT_OBJECT': num_objects += 1
                sym_bind = symbol['st_info']['bind']
                if sym_bind == 'STB_GLOBAL': num_global += 1
                elif sym_bind == 'STB_LOCAL': num_local += 1
                elif sym_bind == 'STB_WEAK': num_weak += 1
                if symbol['st_shndx'] == 'SHN_UNDEF': num_undefined += 1
    
    features['num_symbols'] = num_symbols
    features['num_functions'] = num_functions
    features['num_objects'] = num_objects
    features['num_global_symbols'] = num_global
    features['num_local_symbols'] = num_local
    features['num_weak_symbols'] = num_weak
    features['num_undefined_symbols'] = num_undefined
    
    # Dynamic
    # Dynamic section features:
    # Count dynamic tags and whether important ones exist
    num_needed = num_dynamic_entries = has_rpath = has_runpath = has_soname = 0
    for section in elf.iter_sections():
        if isinstance(section, DynamicSection):
            for tag in section.iter_tags():
                num_dynamic_entries += 1
                if tag.entry.d_tag == 'DT_NEEDED': num_needed += 1
                elif tag.entry.d_tag == 'DT_RPATH': has_rpath = 1
                elif tag.entry.d_tag == 'DT_RUNPATH': has_runpath = 1
                elif tag.entry.d_tag == 'DT_SONAME': has_soname = 1
    
    features['num_libraries_needed'] = num_needed
    features['num_dynamic_entries'] = num_dynamic_entries
    features['has_rpath'] = has_rpath
    features['has_runpath'] = has_runpath
    features['has_soname'] = has_soname
    features['file_size'] = len(data)
    
    # Return as ordered list
    keys = ['e_type', 'e_machine', 'e_version', 'e_entry', 'e_phoff', 'e_shoff',
            'e_flags', 'e_ehsize', 'e_phentsize', 'e_phnum', 'e_shentsize',
            'e_shnum', 'e_shstrndx', 'num_sections', 'sections_executable', 
            'sections_writable', 'sections_allocatable', 'sections_progbits', 
            'sections_nobits', 'sections_symtab', 'sections_dynsym', 'sections_strtab',
            'sections_rela', 'sections_rel', 'sections_dynamic', 'sections_note',
            'section_size_min', 'section_size_max', 'section_size_mean', 'section_size_total',
            'num_segments', 'segments_executable', 'segments_writable', 'segments_readable',
            'segments_load', 'segments_dynamic', 'segments_interp', 'segments_note',
            'segments_gnu_stack', 'segments_gnu_relro', 'num_symbols', 'num_functions',
            'num_objects', 'num_global_symbols', 'num_local_symbols', 'num_weak_symbols',
            'num_undefined_symbols', 'num_libraries_needed', 'num_dynamic_entries',
            'has_rpath', 'has_runpath', 'has_soname', 'file_size']
    return [features.get(k, 0) for k in keys]

# ============================================================
# Pipeline B: Statistical Features (44 features)
# ============================================================

# ------------------------------------------------------------
# windowed_entropy():
# Computes entropy over sliding windows (default 256 bytes, step 128).
# Returns min/max/mean/std of window entropies.
# This helps detect localized packing/encryption regions rather than only global entropy.
# ------------------------------------------------------------

def windowed_entropy(data, window_size=256, step=128):
    if len(data) < window_size:
        ent = entropy(data)
        return ent, ent, ent, 0.0
    entropies = []
    for i in range(0, len(data) - window_size + 1, step):
        entropies.append(entropy(data[i:i + window_size]))
    if not entropies:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(entropies) / len(entropies)
    std = math.sqrt(sum((e - mean) ** 2 for e in entropies) / len(entropies))
    return min(entropies), max(entropies), mean, std

# ------------------------------------------------------------
# extract_pipeline_b():
# Extracts byte-level statistics:
# - Global and windowed entropy
# - 16-bin normalized byte histogram + special byte groups (null/printable/high bytes)
# - N-gram stats for n=2,3,4 (unique count, entropy, most common freq, diversity measure)
# - Header entropy and a few ELF-section entropies (.text/.data/.rodata) when available
# Returns a fixed-order feature vector (44 features).
# ------------------------------------------------------------

def extract_pipeline_b(data: bytes) -> list:
    features = {}
    
    # Global entropy
    features['global_entropy'] = entropy(data)
    
    # Windowed entropy
    e_min, e_max, e_mean, e_std = windowed_entropy(data)
    features['entropy_min'] = e_min
    features['entropy_max'] = e_max
    features['entropy_mean'] = e_mean
    features['entropy_std'] = e_std
    
    # Byte histogram (16 bins)
    counts = Counter(data)
    total = len(data) if data else 1
    for bin_idx in range(16):
        bin_count = sum(counts.get(b, 0) for b in range(bin_idx * 16, (bin_idx + 1) * 16))
        features[f'byte_freq_bin_{bin_idx}'] = bin_count / total
    
    features['null_byte_freq'] = counts.get(0, 0) / total
    features['printable_freq'] = sum(counts.get(b, 0) for b in range(32, 127)) / total
    features['high_byte_freq'] = sum(counts.get(b, 0) for b in range(128, 256)) / total
    
    # N-grams
    for n in [2, 3, 4]:
        if len(data) < n:
            features[f'ngram_{n}_unique'] = 0
            features[f'ngram_{n}_entropy'] = 0.0
            features[f'ngram_{n}_most_common_freq'] = 0.0
            features[f'ngram_{n}_diversity_50'] = 1.0
        else:
            ngrams = [tuple(data[i:i+n]) for i in range(len(data) - n + 1)]
            ngram_counts = Counter(ngrams)
            features[f'ngram_{n}_unique'] = len(ngram_counts)
            total_ng = len(ngrams)
            features[f'ngram_{n}_entropy'] = -sum((c/total_ng) * math.log2(c/total_ng) for c in ngram_counts.values())
            features[f'ngram_{n}_most_common_freq'] = ngram_counts.most_common(1)[0][1] / total_ng if ngram_counts else 0
            sorted_counts = sorted(ngram_counts.values(), reverse=True)
            cumsum = 0
            for i, count in enumerate(sorted_counts):
                cumsum += count
                if cumsum >= total_ng * 0.5:
                    features[f'ngram_{n}_diversity_50'] = (i + 1) / len(sorted_counts)
                    break
            else:
                features[f'ngram_{n}_diversity_50'] = 1.0
    
    # Header features
    header_data = data[:1024] if len(data) > 1024 else data
    features['header_entropy'] = entropy(header_data)
    features['header_null_freq'] = header_data.count(0) / len(header_data) if header_data else 0
    features['elf_header_entropy'] = entropy(data[:64]) if len(data) >= 64 else entropy(data)
    
    # Section entropy
    features['text_section_entropy'] = 0.0
    features['data_section_entropy'] = 0.0
    features['rodata_section_entropy'] = 0.0
    features['bss_section_size'] = 0
    try:
        f = io.BytesIO(data)
        elf = ELFFile(f)
        for section in elf.iter_sections():
            name = section.name
            if section['sh_type'] != 'SHT_NOBITS' and section['sh_size'] > 0:
                try:
                    section_data = section.data()
                    if name == '.text' and section_data:
                        features['text_section_entropy'] = entropy(section_data)
                    elif name == '.data' and section_data:
                        features['data_section_entropy'] = entropy(section_data)
                    elif name == '.rodata' and section_data:
                        features['rodata_section_entropy'] = entropy(section_data)
                except: pass
            if name == '.bss':
                features['bss_section_size'] = section['sh_size']
    except: pass
    
    features['file_size'] = len(data)
    
    keys = ['global_entropy', 'entropy_min', 'entropy_max', 'entropy_mean', 'entropy_std',
            'byte_freq_bin_0', 'byte_freq_bin_1', 'byte_freq_bin_2', 'byte_freq_bin_3',
            'byte_freq_bin_4', 'byte_freq_bin_5', 'byte_freq_bin_6', 'byte_freq_bin_7',
            'byte_freq_bin_8', 'byte_freq_bin_9', 'byte_freq_bin_10', 'byte_freq_bin_11',
            'byte_freq_bin_12', 'byte_freq_bin_13', 'byte_freq_bin_14', 'byte_freq_bin_15',
            'null_byte_freq', 'printable_freq', 'high_byte_freq',
            'ngram_2_unique', 'ngram_2_entropy', 'ngram_2_most_common_freq', 'ngram_2_diversity_50',
            'ngram_3_unique', 'ngram_3_entropy', 'ngram_3_most_common_freq', 'ngram_3_diversity_50',
            'ngram_4_unique', 'ngram_4_entropy', 'ngram_4_most_common_freq', 'ngram_4_diversity_50',
            'header_entropy', 'header_null_freq', 'elf_header_entropy',
            'text_section_entropy', 'data_section_entropy', 'rodata_section_entropy', 'bss_section_size',
            'file_size']
    return [features.get(k, 0) for k in keys]

# ============================================================
# Pipeline C: Hybrid Features (39 features)
# ============================================================

# ------------------------------------------------------------
# extract_strings():
# Simple static string extraction: scan bytes and collect contiguous printable ASCII runs.
# Used to derive features like number of strings, average string length, URL/IP counts.
# ------------------------------------------------------------

def extract_strings(data, min_length=4):
    strings = []
    current = []
    for byte in data:
        if 32 <= byte < 127:
            current.append(chr(byte))
        else:
            if len(current) >= min_length:
                strings.append(''.join(current))
            current = []
    if len(current) >= min_length:
        strings.append(''.join(current))
    return strings

# ------------------------------------------------------------
# is_ip_pattern():
# Lightweight check for IPv4-like patterns inside extracted strings.
# (Heuristic feature: malware sometimes contains hardcoded IPs.)
# ------------------------------------------------------------

def is_ip_pattern(s):
    parts = s.split('.')
    if len(parts) != 4: return False
    try: return all(0 <= int(p) <= 255 for p in parts)
    except: return False

# ------------------------------------------------------------
# extract_pipeline_c():
# Builds a hybrid feature vector:
# - Select a subset from structural features (counts, symbols, needed libs, etc.)
# - Select a subset from statistical features (entropy + a few byte ratios + n-gram entropy)
# - Add new derived/security features:
#   * ratios: code/data, code/file, header/file
#   * section entropy statistics + count of very-high-entropy sections
#   * NX/PIE/RELRO heuristics from segments/header
#   * suspicious imports (execve/socket/ptrace/system...) and string-based URL/IP heuristics
# If parsing fails, return a zero vector to avoid crashing the worker.
# ------------------------------------------------------------

def extract_pipeline_c(data: bytes) -> list:
    features = {}
    
    try:
        f = io.BytesIO(data)
        elf = ELFFile(f)
        
        # From Pipeline A
        features['a_e_type'] = get_value(elf.header['e_type'], E_TYPE_MAP)
        features['a_e_machine'] = get_value(elf.header['e_machine'], E_MACHINE_MAP)
        features['a_num_sections'] = elf.num_sections()
        features['a_num_segments'] = elf.num_segments()
        
        sections_exec = sections_write = 0
        for section in elf.iter_sections():
            flags = section['sh_flags']
            if flags & 0x4: sections_exec += 1
            if flags & 0x1: sections_write += 1
        features['a_sections_executable'] = sections_exec
        features['a_sections_writable'] = sections_write
        
        segments_exec = 0
        for segment in elf.iter_segments():
            if segment['p_flags'] & 0x1: segments_exec += 1
        features['a_segments_executable'] = segments_exec
        
        num_symbols = num_functions = num_undefined = num_needed = 0
        for section in elf.iter_sections():
            if isinstance(section, SymbolTableSection):
                for symbol in section.iter_symbols():
                    num_symbols += 1
                    if symbol['st_info']['type'] == 'STT_FUNC': num_functions += 1
                    if symbol['st_shndx'] == 'SHN_UNDEF': num_undefined += 1
            if isinstance(section, DynamicSection):
                for tag in section.iter_tags():
                    if tag.entry.d_tag == 'DT_NEEDED': num_needed += 1
        
        features['a_num_symbols'] = num_symbols
        features['a_num_functions'] = num_functions
        features['a_num_undefined_symbols'] = num_undefined
        features['a_num_libraries_needed'] = num_needed
        features['a_file_size'] = len(data)
        
        # From Pipeline B
        features['b_global_entropy'] = entropy(data)
        _, _, e_mean, e_std = windowed_entropy(data)
        features['b_entropy_mean'] = e_mean
        features['b_entropy_std'] = e_std
        
        counts = Counter(data)
        total = len(data)
        features['b_null_byte_freq'] = counts.get(0, 0) / total
        features['b_printable_freq'] = sum(counts.get(b, 0) for b in range(32, 127)) / total
        features['b_high_byte_freq'] = sum(counts.get(b, 0) for b in range(128, 256)) / total
        
        for n in [2, 3]:
            ngrams = [tuple(data[i:i+n]) for i in range(len(data) - n + 1)]
            ngram_counts = Counter(ngrams)
            total_ng = len(ngrams)
            features[f'b_ngram_{n}_entropy'] = -sum((c/total_ng) * math.log2(c/total_ng) for c in ngram_counts.values()) if ngram_counts else 0
        
        header_data = data[:1024] if len(data) > 1024 else data
        features['b_header_entropy'] = entropy(header_data)
        
        features['b_text_section_entropy'] = 0.0
        for section in elf.iter_sections():
            if section.name == '.text' and section['sh_type'] != 'SHT_NOBITS':
                try:
                    features['b_text_section_entropy'] = entropy(section.data())
                except: pass
        
        # Derived features (Pipeline C originals)
        text_size = data_size = 0
        for section in elf.iter_sections():
            if section.name == '.text': text_size = section['sh_size']
            elif section.name in ['.data', '.rodata']: data_size += section['sh_size']
        
        features['c_code_to_data_ratio'] = text_size / data_size if data_size > 0 else 0
        features['c_code_to_file_ratio'] = text_size / len(data) if len(data) > 0 else 0
        header_size = elf.header['e_ehsize'] + (elf.header['e_phnum'] * elf.header['e_phentsize'])
        features['c_header_to_file_ratio'] = header_size / len(data) if len(data) > 0 else 0
        
        section_entropies = []
        for section in elf.iter_sections():
            if section['sh_type'] != 'SHT_NOBITS' and section['sh_size'] > 0:
                try:
                    section_entropies.append(entropy(section.data()))
                except: pass
        
        if section_entropies:
            mean_ent = sum(section_entropies) / len(section_entropies)
            features['c_section_entropy_mean'] = mean_ent
            features['c_section_entropy_max'] = max(section_entropies)
            features['c_section_entropy_std'] = math.sqrt(sum((e - mean_ent)**2 for e in section_entropies) / len(section_entropies))
            features['c_high_entropy_section_count'] = sum(1 for e in section_entropies if e > 7.0)
        else:
            features['c_section_entropy_mean'] = features['c_section_entropy_max'] = features['c_section_entropy_std'] = 0
            features['c_high_entropy_section_count'] = 0
        
        # Security features
        has_nx = has_pie = has_relro = 0
        for segment in elf.iter_segments():
            if segment['p_type'] == 'PT_GNU_STACK' and not (segment['p_flags'] & 0x1): has_nx = 1
            elif segment['p_type'] == 'PT_GNU_RELRO': has_relro = 1
        if elf.header['e_type'] == 'ET_DYN':
            for segment in elf.iter_segments():
                if segment['p_type'] == 'PT_INTERP': has_pie = 1; break
        
        features['c_has_nx'] = has_nx
        features['c_has_pie'] = has_pie
        features['c_has_relro'] = has_relro
        
        # Suspicious patterns
        features['c_suspicious_section_count'] = 0
        features['c_empty_section_name_count'] = sum(1 for s in elf.iter_sections() if not s.name.strip())
        
        suspicious_imports = ['execve', 'fork', 'socket', 'connect', 'bind', 'chmod', 'ptrace', 'mprotect', 'dlopen', 'system']
        features['c_suspicious_import_count'] = 0
        for section in elf.iter_sections():
            if isinstance(section, SymbolTableSection):
                for symbol in section.iter_symbols():
                    if any(s in symbol.name.lower() for s in suspicious_imports):
                        features['c_suspicious_import_count'] += 1
        
        strings = extract_strings(data)
        features['c_num_strings'] = len(strings)
        features['c_avg_string_length'] = sum(len(s) for s in strings) / len(strings) if strings else 0
        features['c_url_string_count'] = sum(1 for s in strings if 'http' in s.lower() or 'www.' in s.lower())
        features['c_ip_string_count'] = sum(1 for s in strings if is_ip_pattern(s))
        
    except Exception as e:
        print(f"Error in pipeline_c extraction: {e}")
        return [0] * 39
    
    keys = ['a_e_type', 'a_e_machine', 'a_num_sections', 'a_num_segments',
            'a_sections_executable', 'a_sections_writable', 'a_segments_executable',
            'a_num_symbols', 'a_num_functions', 'a_num_undefined_symbols',
            'a_num_libraries_needed', 'a_file_size',
            'b_global_entropy', 'b_entropy_mean', 'b_entropy_std',
            'b_null_byte_freq', 'b_printable_freq', 'b_high_byte_freq',
            'b_ngram_2_entropy', 'b_ngram_3_entropy',
            'b_header_entropy', 'b_text_section_entropy',
            'c_code_to_data_ratio', 'c_code_to_file_ratio', 'c_header_to_file_ratio',
            'c_section_entropy_mean', 'c_section_entropy_max', 'c_section_entropy_std',
            'c_high_entropy_section_count',
            'c_has_nx', 'c_has_pie', 'c_has_relro',
            'c_suspicious_section_count', 'c_empty_section_name_count',
            'c_suspicious_import_count',
            'c_num_strings', 'c_avg_string_length',
            'c_url_string_count', 'c_ip_string_count']
    return [features.get(k, 0) for k in keys]

# ============================================================
# Process ELF File
# ============================================================

# ------------------------------------------------------------
# process_elf():
# End-to-end inference for a single ELF binary:
# - Extract features for A/B/C.
# - Run RF+XGB predictions for each pipeline and keep label+confidence.
# - Choose final decision from Pipeline C + XGBoost and return a unified result dict.
# Any unexpected error returns an error payload instead of crashing the worker.
# ------------------------------------------------------------

def process_elf(data: bytes) -> dict:
    try:
        # Extract features for all pipelines
        feats_a = extract_pipeline_a(data)
        feats_b = extract_pipeline_b(data)
        feats_c = extract_pipeline_c(data)
        
        results = {
            'file_size': len(data),
            'pipelines': {}
        }
        
        # Pipeline A predictions
        pred_a_rf = int(models['pipeline_a_rf'].predict([feats_a])[0])
        prob_a_rf = float(max(models['pipeline_a_rf'].predict_proba([feats_a])[0]))
        pred_a_xgb = int(models['pipeline_a_xgb'].predict([feats_a])[0])
        prob_a_xgb = float(max(models['pipeline_a_xgb'].predict_proba([feats_a])[0]))
        
        results['pipelines']['A_structural'] = {
            'rf': {'label': 'malicious' if pred_a_rf == 1 else 'benign', 'confidence': round(prob_a_rf, 4)},
            'xgb': {'label': 'malicious' if pred_a_xgb == 1 else 'benign', 'confidence': round(prob_a_xgb, 4)}
        }
        
        # Pipeline B predictions
        pred_b_rf = int(models['pipeline_b_rf'].predict([feats_b])[0])
        prob_b_rf = float(max(models['pipeline_b_rf'].predict_proba([feats_b])[0]))
        pred_b_xgb = int(models['pipeline_b_xgb'].predict([feats_b])[0])
        prob_b_xgb = float(max(models['pipeline_b_xgb'].predict_proba([feats_b])[0]))
        
        results['pipelines']['B_statistical'] = {
            'rf': {'label': 'malicious' if pred_b_rf == 1 else 'benign', 'confidence': round(prob_b_rf, 4)},
            'xgb': {'label': 'malicious' if pred_b_xgb == 1 else 'benign', 'confidence': round(prob_b_xgb, 4)}
        }
        
        # Pipeline C (best) predictions
        pred_c_rf = int(models['pipeline_c_rf'].predict([feats_c])[0])
        prob_c_rf = float(max(models['pipeline_c_rf'].predict_proba([feats_c])[0]))
        pred_c_xgb = int(models['pipeline_c_xgb'].predict([feats_c])[0])
        prob_c_xgb = float(max(models['pipeline_c_xgb'].predict_proba([feats_c])[0]))
        
        results['pipelines']['C_hybrid'] = {
            'rf': {'label': 'malicious' if pred_c_rf == 1 else 'benign', 'confidence': round(prob_c_rf, 4)},
            'xgb': {'label': 'malicious' if pred_c_xgb == 1 else 'benign', 'confidence': round(prob_c_xgb, 4)}
        }
        
        # Final decision: Use Pipeline C XGBoost (best performer)
        results['final_label'] = 'malicious' if pred_c_xgb == 1 else 'benign'
        results['final_confidence'] = round(prob_c_xgb, 4)
        results['best_model'] = 'Pipeline C (Hybrid) + XGBoost'
        
        return results
        
    except Exception as e:
        return {
            'error': str(e),
            'final_label': 'error',
            'final_confidence': 0
        }

# ============================================================
# Main Loop
# ============================================================

# ------------------------------------------------------------
# main_loop():
# Infinite worker loop:
# - BRPOP blocks until a job_id is pushed to the Redis queue.
# - Fetch base64 file content from Redis.
# - Validate ELF magic bytes.
# - Run inference and store results back in Redis as a hash (values JSON strings).
# The API/UI can later read elf:result:{job_id} and display the classification.
# ------------------------------------------------------------

def main_loop():
    print("Worker started, waiting for jobs...")
    
    while True:
        item = redis_client.brpop(QUEUE_KEY)
        if item is None:
            continue
        
        queue_name, job_id = item
        print(f"[worker] Got job_id={job_id}")
        
        file_key = f"elf:file:{job_id}"
        b64_data = redis_client.get(file_key)
        
        if not b64_data:
            print(f"[worker] File for job {job_id} not found")
            continue
        
        data = base64.b64decode(b64_data.encode("utf-8"))
        
        # Check ELF magic
        if data[:4] != b'\x7fELF':
            result = {'error': 'Not a valid ELF file', 'final_label': 'error', 'final_confidence': 0}
        else:
            result = process_elf(data)
        
        result_key = f"elf:result:{job_id}"
        redis_client.hset(result_key,
         mapping={k: json.dumps(v) for k, v in result.items()})
        
        print(f"[worker] Job {job_id} done: {result.get('final_label', 'error')} (conf: {result.get('final_confidence', 0)})")

# ------------------------------------------------------------
# Script entry point:
# Start the worker loop when running this file directly.
# ------------------------------------------------------------

if __name__ == "__main__":
    main_loop()

# ===========================
# Pipeline A — Structural Feature Extraction (ELF-Miner style)
# This file extracts structural and metadata-based features from ELF binaries.
# Extracts structural features from:
# - ELF Header
# - Section Headers
# - Program Headers
# - Symbol Tables
# - Dynamic Linking Information
# ===========================

import math
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from elftools.elf.dynamic import DynamicSection
import io

# --------- Mappings ----------
# Mapping tables that convert categorical 
# ELF fields into numeric values.
# ===========================

E_TYPE_MAP = {
    'ET_NONE': 0, 'ET_REL': 1, 'ET_EXEC': 2, 'ET_DYN': 3, 'ET_CORE': 4,
}

E_MACHINE_MAP = {
    'EM_NONE': 0, 'EM_386': 3, 'EM_ARM': 40, 'EM_X86_64': 62,
    'EM_AARCH64': 183, 'EM_MIPS': 8, 'EM_PPC': 20, 'EM_PPC64': 21,
}

SH_TYPE_MAP = {
    'SHT_NULL': 0, 'SHT_PROGBITS': 1, 'SHT_SYMTAB': 2, 'SHT_STRTAB': 3,
    'SHT_RELA': 4, 'SHT_HASH': 5, 'SHT_DYNAMIC': 6, 'SHT_NOTE': 7,
    'SHT_NOBITS': 8, 'SHT_REL': 9, 'SHT_DYNSYM': 11, 'SHT_INIT_ARRAY': 14,
    'SHT_FINI_ARRAY': 15, 'SHT_GNU_HASH': 0x6ffffff6, 'SHT_GNU_versym': 0x6fffffff,
}

PT_TYPE_MAP = {
    'PT_NULL': 0, 'PT_LOAD': 1, 'PT_DYNAMIC': 2, 'PT_INTERP': 3,
    'PT_NOTE': 4, 'PT_SHLIB': 5, 'PT_PHDR': 6, 'PT_TLS': 7,
    'PT_GNU_EH_FRAME': 0x6474e550, 'PT_GNU_STACK': 0x6474e551,
    'PT_GNU_RELRO': 0x6474e552,
}

# ===========================
# Helper function that converts raw ELF values into numeric form.
# Handles strings, enums, and numeric fields in a uniform way.
# ===========================

def get_value(raw, mapping):
    """Convert string or enum to numeric value."""
    if isinstance(raw, str):
        return mapping.get(raw, hash(raw) % 1000)
    elif hasattr(raw, 'value'):
        return raw.value
    else:
        return int(raw) if raw else 0

# ===========================
# Main feature extraction function for Pipeline A.
# Receives raw ELF bytes and returns a dictionary of structural features.
# ===========================

def extract_pipeline_a(data: bytes) -> dict:

    """
    Extract structural features from ELF file.
    Returns a dictionary of features.
    """

    # Wrap raw bytes as file-like object
    f = io.BytesIO(data)
    elf = ELFFile(f)
    header = elf.header
    
    features = {}
    
    # ============ ELF Header Features ============
    # ELF HEADER FEATURES
    # Capture general metadata about file type, architecture and layout.
    # ===========================
    features['e_type'] = get_value(header['e_type'], E_TYPE_MAP)
    features['e_machine'] = get_value(header['e_machine'], E_MACHINE_MAP)

    # Normalize version field (can be string or int)
    e_version = header['e_version']
    if isinstance(e_version, str):
        features['e_version'] = 1 if 'CURRENT' in e_version else 0
    else:
        features['e_version'] = int(e_version) if e_version else 0
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
    
    # ============ Section Header Features ============
    # SECTION HEADER FEATURES
    # Analyze number, types, permissions, and sizes of ELF sections.
    # ===========================
    n_sections = elf.num_sections()
    features['num_sections'] = n_sections
    
    # Section type counts
    section_types = {}
    section_sizes = []
    section_flags_exec = 0
    section_flags_write = 0
    section_flags_alloc = 0
    
    for section in elf.iter_sections():
        sh_type = section['sh_type']
        type_name = sh_type if isinstance(sh_type, str) else f"TYPE_{sh_type}"
        section_types[type_name] = section_types.get(type_name, 0) + 1
        section_sizes.append(section['sh_size'])
        
        flags = section['sh_flags']
        if flags & 0x1:  # SHF_WRITE
            section_flags_write += 1
        if flags & 0x2:  # SHF_ALLOC
            section_flags_alloc += 1
        if flags & 0x4:  # SHF_EXECINSTR
            section_flags_exec += 1
    
    features['sections_executable'] = section_flags_exec
    features['sections_writable'] = section_flags_write
    features['sections_allocatable'] = section_flags_alloc
    
    # Section type specific counts
    features['sections_progbits'] = section_types.get('SHT_PROGBITS', 0)
    features['sections_nobits'] = section_types.get('SHT_NOBITS', 0)
    features['sections_symtab'] = section_types.get('SHT_SYMTAB', 0)
    features['sections_dynsym'] = section_types.get('SHT_DYNSYM', 0)
    features['sections_strtab'] = section_types.get('SHT_STRTAB', 0)
    features['sections_rela'] = section_types.get('SHT_RELA', 0)
    features['sections_rel'] = section_types.get('SHT_REL', 0)
    features['sections_dynamic'] = section_types.get('SHT_DYNAMIC', 0)
    features['sections_note'] = section_types.get('SHT_NOTE', 0)
    
    # Section size statistics
    if section_sizes:
        features['section_size_min'] = min(section_sizes)
        features['section_size_max'] = max(section_sizes)
        features['section_size_mean'] = sum(section_sizes) / len(section_sizes)
        features['section_size_total'] = sum(section_sizes)
    else:
        features['section_size_min'] = 0
        features['section_size_max'] = 0
        features['section_size_mean'] = 0
        features['section_size_total'] = 0
    
    # ============ Program Header Features ============
    # PROGRAM HEADER (SEGMENT) FEATURES
    # Segments describe how the program is loaded into memory.
    # ===========================

    n_segments = elf.num_segments()
    features['num_segments'] = n_segments
    
    segment_types = {}
    segment_flags_exec = 0
    segment_flags_write = 0
    segment_flags_read = 0
    
    for segment in elf.iter_segments():
        p_type = segment['p_type']
        type_name = p_type if isinstance(p_type, str) else f"TYPE_{p_type}"
        segment_types[type_name] = segment_types.get(type_name, 0) + 1
        
        flags = segment['p_flags']
        if flags & 0x1:  # PF_X
            segment_flags_exec += 1
        if flags & 0x2:  # PF_W
            segment_flags_write += 1
        if flags & 0x4:  # PF_R
            segment_flags_read += 1
    
    features['segments_executable'] = segment_flags_exec
    features['segments_writable'] = segment_flags_write
    features['segments_readable'] = segment_flags_read
    
    # Segment type specific counts
    features['segments_load'] = segment_types.get('PT_LOAD', 0)
    features['segments_dynamic'] = segment_types.get('PT_DYNAMIC', 0)
    features['segments_interp'] = segment_types.get('PT_INTERP', 0)
    features['segments_note'] = segment_types.get('PT_NOTE', 0)
    features['segments_gnu_stack'] = segment_types.get('PT_GNU_STACK', 0)
    features['segments_gnu_relro'] = segment_types.get('PT_GNU_RELRO', 0)
    
    # ============ Symbol Table Features ============
    # SYMBOL TABLE FEATURES
    # Count functions, objects, bindings, and undefined references.
    # ===========================
    num_symbols = 0
    num_functions = 0
    num_objects = 0
    num_global = 0
    num_local = 0
    num_weak = 0
    num_undefined = 0
    
    for section in elf.iter_sections():
        if isinstance(section, SymbolTableSection):
            for symbol in section.iter_symbols():
                num_symbols += 1
                
                # Symbol type
                sym_type = symbol['st_info']['type']
                if sym_type == 'STT_FUNC':
                    num_functions += 1
                elif sym_type == 'STT_OBJECT':
                    num_objects += 1
                
                # Symbol binding
                sym_bind = symbol['st_info']['bind']
                if sym_bind == 'STB_GLOBAL':
                    num_global += 1
                elif sym_bind == 'STB_LOCAL':
                    num_local += 1
                elif sym_bind == 'STB_WEAK':
                    num_weak += 1
                
                # Undefined symbols
                if symbol['st_shndx'] == 'SHN_UNDEF':
                    num_undefined += 1
    
    features['num_symbols'] = num_symbols
    features['num_functions'] = num_functions
    features['num_objects'] = num_objects
    features['num_global_symbols'] = num_global
    features['num_local_symbols'] = num_local
    features['num_weak_symbols'] = num_weak
    features['num_undefined_symbols'] = num_undefined
    
    # ============ Dynamic Section Features ============
    # DYNAMIC LINKING FEATURES
    # Describe shared library dependencies and runtime loader behavior.
    # ===========================

    num_needed = 0  
    num_dynamic_entries = 0
    has_rpath = 0
    has_runpath = 0
    has_soname = 0
    
    for section in elf.iter_sections():
        if isinstance(section, DynamicSection):
            for tag in section.iter_tags():
                num_dynamic_entries += 1
                
                if tag.entry.d_tag == 'DT_NEEDED':
                    num_needed += 1
                elif tag.entry.d_tag == 'DT_RPATH':
                    has_rpath = 1
                elif tag.entry.d_tag == 'DT_RUNPATH':
                    has_runpath = 1
                elif tag.entry.d_tag == 'DT_SONAME':
                    has_soname = 1
    
    features['num_libraries_needed'] = num_needed
    features['num_dynamic_entries'] = num_dynamic_entries
    features['has_rpath'] = has_rpath
    features['has_runpath'] = has_runpath
    features['has_soname'] = has_soname
    
    # ============ File Size ============
    features['file_size'] = len(data)
    
    return features

# ===========================
# Returns the fixed feature order expected by the ML models.
# ===========================

def get_pipeline_a_feature_names():
    """Return list of feature names in order."""
    return [
        # ELF Header
        'e_type', 'e_machine', 'e_version', 'e_entry', 'e_phoff', 'e_shoff',
        'e_flags', 'e_ehsize', 'e_phentsize', 'e_phnum', 'e_shentsize',
        'e_shnum', 'e_shstrndx',
        # Section counts
        'num_sections', 'sections_executable', 'sections_writable',
        'sections_allocatable', 'sections_progbits', 'sections_nobits',
        'sections_symtab', 'sections_dynsym', 'sections_strtab',
        'sections_rela', 'sections_rel', 'sections_dynamic', 'sections_note',
        # Section sizes
        'section_size_min', 'section_size_max', 'section_size_mean', 'section_size_total',
        # Segment counts
        'num_segments', 'segments_executable', 'segments_writable',
        'segments_readable', 'segments_load', 'segments_dynamic',
        'segments_interp', 'segments_note', 'segments_gnu_stack', 'segments_gnu_relro',
        # Symbols
        'num_symbols', 'num_functions', 'num_objects', 'num_global_symbols',
        'num_local_symbols', 'num_weak_symbols', 'num_undefined_symbols',
        # Dynamic
        'num_libraries_needed', 'num_dynamic_entries', 'has_rpath',
        'has_runpath', 'has_soname',
        # Size
        'file_size',
    ]


# ===========================
# Converts feature dictionary into ordered vector for ML inference.
# ===========================

def extract_pipeline_a_vector(data: bytes) -> list:
    """Extract features as a vector (list) for ML."""
    features = extract_pipeline_a(data)
    names = get_pipeline_a_feature_names()
    return [features.get(name, 0) for name in names]


# ===========================
# Local test entry-point:
# Allows manual testing of feature extraction from command line.
# ===========================

if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            data = f.read()
        features = extract_pipeline_a(data)
        print(f"Extracted {len(features)} features:")
        for k, v in features.items():
            print(f"  {k}: {v}")

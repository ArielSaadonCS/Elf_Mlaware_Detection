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

import math  # imported earlier for potential statistical extensions (kept for compatibility)
from elftools.elf.elffile import ELFFile  # primary parser for ELF binaries
from elftools.elf.sections import SymbolTableSection  # type check for symbol-table sections
from elftools.elf.dynamic import DynamicSection  # type check for .dynamic sections
import io  # used to wrap bytes into a file-like object for ELFFile

# --------- Mappings ----------
# Mapping tables that convert categorical 
# ELF fields into numeric values. These maps let downstream ML
# models consume categorical metadata as numeric inputs.
# ===========================

E_TYPE_MAP = {
    'ET_NONE': 0, 'ET_REL': 1, 'ET_EXEC': 2, 'ET_DYN': 3, 'ET_CORE': 4,
}
# Mapping of e_type values (file type) to integers.

E_MACHINE_MAP = {
    'EM_NONE': 0, 'EM_386': 3, 'EM_ARM': 40, 'EM_X86_64': 62,
    'EM_AARCH64': 183, 'EM_MIPS': 8, 'EM_PPC': 20, 'EM_PPC64': 21,
}
# Mapping of e_machine values (architecture) to integers.

SH_TYPE_MAP = {
    'SHT_NULL': 0, 'SHT_PROGBITS': 1, 'SHT_SYMTAB': 2, 'SHT_STRTAB': 3,
    'SHT_RELA': 4, 'SHT_HASH': 5, 'SHT_DYNAMIC': 6, 'SHT_NOTE': 7,
    'SHT_NOBITS': 8, 'SHT_REL': 9, 'SHT_DYNSYM': 11, 'SHT_INIT_ARRAY': 14,
    'SHT_FINI_ARRAY': 15, 'SHT_GNU_HASH': 0x6ffffff6, 'SHT_GNU_versym': 0x6fffffff,
}
# Known section types mapped to numeric constants. Not used directly everywhere,
# but kept for completeness and if conversion via get_value is needed.

PT_TYPE_MAP = {
    'PT_NULL': 0, 'PT_LOAD': 1, 'PT_DYNAMIC': 2, 'PT_INTERP': 3,
    'PT_NOTE': 4, 'PT_SHLIB': 5, 'PT_PHDR': 6, 'PT_TLS': 7,
    'PT_GNU_EH_FRAME': 0x6474e550, 'PT_GNU_STACK': 0x6474e551,
    'PT_GNU_RELRO': 0x6474e552,
}
# Known program header (segment) types mapped to numeric constants.

# ===========================
# Helper function that converts raw ELF values into numeric form.
# Handles strings, enums, and numeric fields in a uniform way.
# ===========================

def get_value(raw, mapping):
    """
    Convert a raw ELF field to a numeric value suitable for ML features.

    Parameters:
    - raw: The raw value extracted from an ELF header/field. This may be:
      - a Python string (e.g., 'ET_EXEC' or 'SHT_PROGBITS'),
      - an enum-like object with a .value attribute,
      - or a numeric type (int) already.
    - mapping: A dictionary mapping string names to integers. If 'raw' is a string
      the mapping will be consulted to get a stable numeric code.

    Returns:
    - int: numeric representation of `raw`. The function uses these rules:
      1. If raw is a string: return mapping.get(raw, fallback) where fallback is
         hash(raw) % 1000 to produce a deterministic but bounded integer.
      2. Else, if raw has a .value attribute (enum from pyelftools), return that.
      3. Else, attempt to cast raw to int; if raw is falsy (None/0), return 0.
    """
    # If the value is already a human-readable string (like 'ET_EXEC'), map it.
    if isinstance(raw, str):
        # Use mapping lookup; if missing, compute a fallback numeric fingerprint.
        return mapping.get(raw, hash(raw) % 1000)
    # If the object is an enum-like with a numeric .value, use that numeric value.
    elif hasattr(raw, 'value'):
        return raw.value
    # Otherwise, attempt to cast to int. If raw is falsy (None or 0) return 0.
    else:
        return int(raw) if raw else 0

# ===========================
# Main feature extraction function for Pipeline A.
# Receives raw ELF bytes and returns a dictionary of structural features.
# ===========================

def extract_pipeline_a(data: bytes) -> dict:
    """
    Extract structural features from an ELF file (Pipeline A).

    Parameters:
    - data: bytes
        Raw bytes of the ELF file to analyze.

    Purpose:
    - Parse the ELF file using pyelftools and extract a fixed set of structural
      and metadata-based features (header fields, section/segment counts and
      properties, symbol table summary, dynamic linking info, file size) and
      return them in a dictionary keyed by feature names.

    Returns:
    - dict
        A dictionary mapping feature names (strings) to numeric values.
        Missing or inapplicable features are present with 0 (or appropriate default).
    """
    # Wrap raw bytes as a file-like object so pyelftools can parse it.
    f = io.BytesIO(data)
    # Parse the ELF file structure and store the ELFFile instance.
    elf = ELFFile(f)
    # Extract the top-level ELF header dictionary for easy access.
    header = elf.header

    # Initialize the features dictionary that will be returned.
    features = {}

    # ============ ELF Header Features ============
    # ELF HEADER FEATURES — Capture general metadata about file type, architecture and layout.
    # Use get_value to convert enum/string fields to numeric codes.

    # Convert e_type (file type) into numeric using mapping.
    features['e_type'] = get_value(header['e_type'], E_TYPE_MAP)
    # Convert e_machine (architecture) into numeric using mapping.
    features['e_machine'] = get_value(header['e_machine'], E_MACHINE_MAP)

    # Normalize version field (can be string or int)
    e_version = header['e_version']
    # If e_version is string-like, check if it contains 'CURRENT' and map to 1/0.
    if isinstance(e_version, str):
        features['e_version'] = 1 if 'CURRENT' in e_version else 0
    else:
        # If numeric or enum-like, cast to int, default to 0 on falsy values.
        features['e_version'] = int(e_version) if e_version else 0

    # Capture numeric header fields directly. These come from pyelftools header mapping.
    features['e_entry'] = header['e_entry']       # entry point virtual address
    features['e_phoff'] = header['e_phoff']       # program header table file offset
    features['e_shoff'] = header['e_shoff']       # section header table file offset
    features['e_flags'] = header['e_flags']       # processor-specific flags
    features['e_ehsize'] = header['e_ehsize']     # ELF header size in bytes
    features['e_phentsize'] = header['e_phentsize']  # size of one program header entry
    features['e_phnum'] = header['e_phnum']       # number of program header entries
    features['e_shentsize'] = header['e_shentsize']  # size of one section header entry
    features['e_shnum'] = header['e_shnum']       # number of section header entries
    features['e_shstrndx'] = header['e_shstrndx'] # index of section name string table

    # ============ Section Header Features ============
    # SECTION HEADER FEATURES — Analyze number, types, permissions, and sizes of ELF sections.

    # Number of sections in the ELF
    n_sections = elf.num_sections()
    features['num_sections'] = n_sections

    # Prepare structures to accumulate per-section statistics.
    section_types = {}         # counts of each section type by textual name
    section_sizes = []         # list of sizes of each section (for stats)
    section_flags_exec = 0     # count of sections with executable bit set
    section_flags_write = 0    # count of sections with writable bit set
    section_flags_alloc = 0    # count of sections with allocatable bit set

    # Iterate all sections to compute the above aggregates.
    for section in elf.iter_sections():
        # Extract the numeric/textual sh_type field
        sh_type = section['sh_type']
        # If pyelftools returns a string like "SHT_PROGBITS", use it; otherwise create synthetic name.
        type_name = sh_type if isinstance(sh_type, str) else f"TYPE_{sh_type}"
        # Increment count for this section type name.
        section_types[type_name] = section_types.get(type_name, 0) + 1
        # Append the section's size for later statistics.
        section_sizes.append(section['sh_size'])

        # Flags field is a bitmask; check standard section flags by bit positions.
        flags = section['sh_flags']
        # SHF_WRITE is usually bit 0x1: if present, section is writable.
        if flags & 0x1:  # SHF_WRITE
            section_flags_write += 1
        # SHF_ALLOC (0x2) indicates section is loaded into memory.
        if flags & 0x2:  # SHF_ALLOC
            section_flags_alloc += 1
        # SHF_EXECINSTR (0x4) indicates the section contains executable instructions.
        if flags & 0x4:  # SHF_EXECINSTR
            section_flags_exec += 1

    # Store aggregated section flag counts into features dictionary.
    features['sections_executable'] = section_flags_exec
    features['sections_writable'] = section_flags_write
    features['sections_allocatable'] = section_flags_alloc

    # Store counts for frequently used section types (default to 0 if absent).
    features['sections_progbits'] = section_types.get('SHT_PROGBITS', 0)
    features['sections_nobits'] = section_types.get('SHT_NOBITS', 0)
    features['sections_symtab'] = section_types.get('SHT_SYMTAB', 0)
    features['sections_dynsym'] = section_types.get('SHT_DYNSYM', 0)
    features['sections_strtab'] = section_types.get('SHT_STRTAB', 0)
    features['sections_rela'] = section_types.get('SHT_RELA', 0)
    features['sections_rel'] = section_types.get('SHT_REL', 0)
    features['sections_dynamic'] = section_types.get('SHT_DYNAMIC', 0)
    features['sections_note'] = section_types.get('SHT_NOTE', 0)

    # Compute basic statistics over section sizes (min, max, mean, total).
    if section_sizes:
        features['section_size_min'] = min(section_sizes)                 # smallest section
        features['section_size_max'] = max(section_sizes)                 # largest section
        features['section_size_mean'] = sum(section_sizes) / len(section_sizes)  # average size
        features['section_size_total'] = sum(section_sizes)               # total of all sections sizes
    else:
        # Default zeros when no sections are present.
        features['section_size_min'] = 0
        features['section_size_max'] = 0
        features['section_size_mean'] = 0
        features['section_size_total'] = 0

    # ============ Program Header Features ============
    # PROGRAM HEADER (SEGMENT) FEATURES — Segments describe how the program is loaded into memory.

    # Number of program segments
    n_segments = elf.num_segments()
    features['num_segments'] = n_segments

    # Prepare accumulators for segment types and permission flags.
    segment_types = {}
    segment_flags_exec = 0
    segment_flags_write = 0
    segment_flags_read = 0

    # Iterate through program headers (segments).
    for segment in elf.iter_segments():
        # p_type may be a string (e.g., 'PT_LOAD') or numeric; form a comparable key.
        p_type = segment['p_type']
        type_name = p_type if isinstance(p_type, str) else f"TYPE_{p_type}"
        # Count this segment type.
        segment_types[type_name] = segment_types.get(type_name, 0) + 1

        # p_flags is a bitmask; test bits for execute/read/write.
        flags = segment['p_flags']
        # PF_X typically 0x1 indicates executable segment.
        if flags & 0x1:  # PF_X
            segment_flags_exec += 1
        # PF_W typically 0x2 indicates writable segment.
        if flags & 0x2:  # PF_W
            segment_flags_write += 1
        # PF_R typically 0x4 indicates readable segment.
        if flags & 0x4:  # PF_R
            segment_flags_read += 1

    # Store aggregated segment flags in features.
    features['segments_executable'] = segment_flags_exec
    features['segments_writable'] = segment_flags_write
    features['segments_readable'] = segment_flags_read

    # Store counts for common segment types (0 if absent).
    features['segments_load'] = segment_types.get('PT_LOAD', 0)
    features['segments_dynamic'] = segment_types.get('PT_DYNAMIC', 0)
    features['segments_interp'] = segment_types.get('PT_INTERP', 0)
    features['segments_note'] = segment_types.get('PT_NOTE', 0)
    features['segments_gnu_stack'] = segment_types.get('PT_GNU_STACK', 0)
    features['segments_gnu_relro'] = segment_types.get('PT_GNU_RELRO', 0)

    # ============ Symbol Table Features ============
    # SYMBOL TABLE FEATURES — Count functions, objects, bindings, and undefined references.

    # Initialize counters for symbols and symbol subtypes.
    num_symbols = 0
    num_functions = 0
    num_objects = 0
    num_global = 0
    num_local = 0
    num_weak = 0
    num_undefined = 0

    # Iterate sections again to find symbol tables (both .symtab and .dynsym are SymbolTableSection instances).
    for section in elf.iter_sections():
        # Only process sections that are real symbol tables according to pyelftools.
        if isinstance(section, SymbolTableSection):
            # Iterate every symbol in the symbol table section.
            for symbol in section.iter_symbols():
                num_symbols += 1  # increment total symbol count

                # Determine symbol type from st_info.type (string or enum-like)
                sym_type = symbol['st_info']['type']
                # If the symbol represents a function, increment function count.
                if sym_type == 'STT_FUNC':
                    num_functions += 1
                # If the symbol represents an object (data), increment object count.
                elif sym_type == 'STT_OBJECT':
                    num_objects += 1

                # Determine symbol binding from st_info.bind
                sym_bind = symbol['st_info']['bind']
                if sym_bind == 'STB_GLOBAL':
                    num_global += 1
                elif sym_bind == 'STB_LOCAL':
                    num_local += 1
                elif sym_bind == 'STB_WEAK':
                    num_weak += 1

                # If symbol's section index is SHN_UNDEF, the symbol is undefined (external reference).
                if symbol['st_shndx'] == 'SHN_UNDEF':
                    num_undefined += 1

    # Store symbol-related features in the features dictionary.
    features['num_symbols'] = num_symbols
    features['num_functions'] = num_functions
    features['num_objects'] = num_objects
    features['num_global_symbols'] = num_global
    features['num_local_symbols'] = num_local
    features['num_weak_symbols'] = num_weak
    features['num_undefined_symbols'] = num_undefined

    # ============ Dynamic Section Features ============
    # DYNAMIC LINKING FEATURES — Describe shared library dependencies and runtime loader behavior.

    # Initialize counters and flags for dynamic section metadata.
    num_needed = 0              # number of DT_NEEDED entries (shared library dependencies)
    num_dynamic_entries = 0     # total number of dynamic tags found
    has_rpath = 0               # flag: presence of DT_RPATH
    has_runpath = 0             # flag: presence of DT_RUNPATH
    has_soname = 0              # flag: presence of DT_SONAME

    # Iterate sections to find the DynamicSection (.dynamic)
    for section in elf.iter_sections():
        if isinstance(section, DynamicSection):
            # Iterate tags inside the dynamic section
            for tag in section.iter_tags():
                num_dynamic_entries += 1  # increment dynamic tag count

                # Tag entries are often enums or constants; compare textual d_tag if available.
                if tag.entry.d_tag == 'DT_NEEDED':
                    num_needed += 1
                elif tag.entry.d_tag == 'DT_RPATH':
                    has_rpath = 1
                elif tag.entry.d_tag == 'DT_RUNPATH':
                    has_runpath = 1
                elif tag.entry.d_tag == 'DT_SONAME':
                    has_soname = 1

    # Store dynamic-related features.
    features['num_libraries_needed'] = num_needed
    features['num_dynamic_entries'] = num_dynamic_entries
    features['has_rpath'] = has_rpath
    features['has_runpath'] = has_runpath
    features['has_soname'] = has_soname

    # ============ File Size ============
    # Capture the raw file size in bytes.
    features['file_size'] = len(data)

    # Return the complete features dictionary.
    return features

# ===========================
# Returns the fixed feature order expected by the ML models.
# ===========================

def get_pipeline_a_feature_names():
    """
    Return the ordered list of feature names expected by downstream ML models.

    Purpose:
    - Provides a stable ordering so that extract_pipeline_a_vector can convert the
      features dictionary into a consistent vector layout for model input.

    Returns:
    - list[str]: feature names in the exact order used by extract_pipeline_a_vector.
    """
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
    """
    Extract features and return them as an ordered numeric vector.

    Parameters:
    - data: bytes
        Raw ELF bytes to analyze.

    Purpose:
    - Calls extract_pipeline_a to obtain a dictionary of features, then orders
      the values according to get_pipeline_a_feature_names to produce a list
      suitable for ML model input (fixed ordering).

    Returns:
    - list[int|float]: ordered list of feature values. Missing dictionary keys
      are filled as 0 via features.get(name, 0).
    """
    # First, extract the features dictionary using the main extractor.
    features = extract_pipeline_a(data)
    # Fetch the canonical ordering of names.
    names = get_pipeline_a_feature_names()
    # Build and return the vector (list) with default 0 for any missing feature.
    return [features.get(name, 0) for name in names]


# ===========================
# Local test entry-point:
# Allows manual testing of feature extraction from command line.
# ===========================

if __name__ == "__main__":
    # Test: allow running this file as a script to extract features from a local ELF file path.
    import sys  # local import to keep top-level imports minimal
    # If a file path was passed on the command line, read and analyze it.
    if len(sys.argv) > 1:
        # Open specified path in binary mode and read bytes.
        with open(sys.argv[1], 'rb') as f:
            data = f.read()
        # Extract features dictionary for the provided file bytes.
        features = extract_pipeline_a(data)
        # Print a short summary: number of extracted feature keys and each key/value.
        print(f"Extracted {len(features)} features:")
        for k, v in features.items():
            print(f"  {k}: {v}")

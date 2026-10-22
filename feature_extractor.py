"""
MalwareGuard AI — Static PE Feature Extraction

Extracts features from a Windows PE (Portable Executable) file for malware
classification:
  - PE header attributes
  - File entropy
  - Section-level statistics
  - Import / Export table counts
"""

import math
import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import pefile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _byte_entropy(data: bytes) -> float:
    """Shannon entropy of a byte sequence (0–8 bits per byte)."""
    if not data:
        return 0.0
    freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    prob = freq[freq > 0] / len(data)
    return float(-np.sum(prob * np.log2(prob)))


def _safe_get(pe_obj, attr: str, default=0):
    try:
        return getattr(pe_obj, attr, default) or default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Feature names (must stay in sync with extraction order)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # File-level
    "file_size",
    "file_entropy",
    "md5_numeric",          # first 8 hex chars → int (weak uniqueness signal)

    # DOS / COFF header
    "e_magic",
    "machine",
    "number_of_sections",
    "timestamp",
    "pointer_to_symbol_table",
    "number_of_symbols",
    "size_of_optional_header",
    "characteristics",

    # Optional header
    "magic",
    "major_linker_version",
    "minor_linker_version",
    "size_of_code",
    "size_of_initialized_data",
    "size_of_uninitialized_data",
    "address_of_entry_point",
    "base_of_code",
    "image_base",
    "section_alignment",
    "file_alignment",
    "major_os_version",
    "minor_os_version",
    "major_image_version",
    "minor_image_version",
    "major_subsystem_version",
    "minor_subsystem_version",
    "size_of_image",
    "size_of_headers",
    "checksum",
    "subsystem",
    "dll_characteristics",
    "size_of_stack_reserve",
    "size_of_stack_commit",
    "size_of_heap_reserve",
    "size_of_heap_commit",
    "number_of_rva_and_sizes",

    # Import / Export
    "number_of_imported_dlls",
    "number_of_imported_functions",
    "number_of_exported_functions",

    # Section aggregates
    "section_max_entropy",
    "section_mean_entropy",
    "section_min_entropy",
    "section_mean_raw_size",
    "section_max_raw_size",
    "section_total_raw_size",
    "section_mean_virtual_size",
    "section_max_virtual_size",
    "section_executable_count",
    "section_writable_count",
    "section_readable_count",
]

NUM_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(file_path: Union[str, Path]) -> np.ndarray:
    """
    Parse a PE file and return a 1-D float numpy array of shape (NUM_FEATURES,).

    Raises
    ------
    ValueError
        If the file cannot be parsed as a valid PE.
    """
    path = Path(file_path)
    raw = path.read_bytes()

    try:
        pe = pefile.PE(data=raw, fast_load=False)
    except pefile.PEFormatError as exc:
        raise ValueError(f"Not a valid PE file: {exc}") from exc

    feat: list = []

    # --- File-level ---
    feat.append(len(raw))
    feat.append(_byte_entropy(raw))
    feat.append(int(hashlib.md5(raw).hexdigest()[:8], 16))

    # --- DOS / COFF header ---
    dh = pe.DOS_HEADER
    fh = pe.FILE_HEADER
    feat.extend([
        _safe_get(dh, "e_magic"),
        _safe_get(fh, "Machine"),
        _safe_get(fh, "NumberOfSections"),
        _safe_get(fh, "TimeDateStamp"),
        _safe_get(fh, "PointerToSymbolTable"),
        _safe_get(fh, "NumberOfSymbols"),
        _safe_get(fh, "SizeOfOptionalHeader"),
        _safe_get(fh, "Characteristics"),
    ])

    # --- Optional header ---
    oh = pe.OPTIONAL_HEADER
    feat.extend([
        _safe_get(oh, "Magic"),
        _safe_get(oh, "MajorLinkerVersion"),
        _safe_get(oh, "MinorLinkerVersion"),
        _safe_get(oh, "SizeOfCode"),
        _safe_get(oh, "SizeOfInitializedData"),
        _safe_get(oh, "SizeOfUninitializedData"),
        _safe_get(oh, "AddressOfEntryPoint"),
        _safe_get(oh, "BaseOfCode"),
        _safe_get(oh, "ImageBase"),
        _safe_get(oh, "SectionAlignment"),
        _safe_get(oh, "FileAlignment"),
        _safe_get(oh, "MajorOperatingSystemVersion"),
        _safe_get(oh, "MinorOperatingSystemVersion"),
        _safe_get(oh, "MajorImageVersion"),
        _safe_get(oh, "MinorImageVersion"),
        _safe_get(oh, "MajorSubsystemVersion"),
        _safe_get(oh, "MinorSubsystemVersion"),
        _safe_get(oh, "SizeOfImage"),
        _safe_get(oh, "SizeOfHeaders"),
        _safe_get(oh, "CheckSum"),
        _safe_get(oh, "Subsystem"),
        _safe_get(oh, "DllCharacteristics"),
        _safe_get(oh, "SizeOfStackReserve"),
        _safe_get(oh, "SizeOfStackCommit"),
        _safe_get(oh, "SizeOfHeapReserve"),
        _safe_get(oh, "SizeOfHeapCommit"),
        _safe_get(oh, "NumberOfRvaAndSizes"),
    ])

    # --- Imports ---
    n_dlls = 0
    n_imports = 0
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            n_dlls += 1
            n_imports += len(entry.imports)
    feat.extend([n_dlls, n_imports])

    # --- Exports ---
    n_exports = 0
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        n_exports = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    feat.append(n_exports)

    # --- Sections ---
    section_entropies = []
    section_raw_sizes = []
    section_virt_sizes = []
    exec_count = writable_count = readable_count = 0

    for sec in pe.sections:
        sec_data = sec.get_data()
        section_entropies.append(_byte_entropy(sec_data))
        section_raw_sizes.append(_safe_get(sec, "SizeOfRawData"))
        section_virt_sizes.append(_safe_get(sec, "Misc_VirtualSize"))
        chars = _safe_get(sec, "Characteristics")
        if chars & 0x20000000:
            exec_count += 1
        if chars & 0x80000000:
            writable_count += 1
        if chars & 0x40000000:
            readable_count += 1

    def _arr_stat(arr, fn):
        return fn(arr) if arr else 0.0

    feat.extend([
        _arr_stat(section_entropies, max),
        _arr_stat(section_entropies, lambda a: sum(a) / len(a)),
        _arr_stat(section_entropies, min),
        _arr_stat(section_raw_sizes, lambda a: sum(a) / len(a)),
        _arr_stat(section_raw_sizes, max),
        sum(section_raw_sizes),
        _arr_stat(section_virt_sizes, lambda a: sum(a) / len(a)),
        _arr_stat(section_virt_sizes, max),
        exec_count,
        writable_count,
        readable_count,
    ])

    pe.close()
    return np.array(feat, dtype=np.float64)

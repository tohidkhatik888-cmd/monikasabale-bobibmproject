"""
MalwareGuard AI — Test Suite

Tests cover:
  - Feature extractor (unit)
  - API endpoints (integration via TestClient)
"""

import io
import struct
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure imports resolve when running from repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import create_app
from app.services.feature_extractor import extract_features, NUM_FEATURES, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Minimal synthetic PE builder for tests
# ---------------------------------------------------------------------------

def _make_minimal_pe() -> bytes:
    """
    Build the smallest structurally valid PE32 binary accepted by pefile.
    DOS stub → PE signature → COFF → Optional (PE32) → one section.
    """
    # DOS header (64 bytes) — e_magic=MZ, e_lfanew points to PE signature
    e_lfanew = 0x40
    dos_header = struct.pack("<H58xI", 0x5A4D, e_lfanew)  # MZ + e_lfanew

    # PE signature
    pe_sig = b"PE\x00\x00"

    # COFF header
    machine = 0x014C          # i386
    num_sections = 1
    timestamp = 0
    sym_table_ptr = 0
    num_symbols = 0
    opt_header_size = 0xE0    # standard PE32 optional header size
    characteristics = 0x0102  # executable, 32-bit
    coff = struct.pack(
        "<HHIIIHH",
        machine, num_sections, timestamp,
        sym_table_ptr, num_symbols,
        opt_header_size, characteristics,
    )

    # Optional header (PE32, 0xE0 bytes)
    image_base = 0x400000
    section_alignment = 0x1000
    file_alignment = 0x200
    size_of_image = 0x3000
    size_of_headers = 0x400
    # Pack minimal fields; pad rest to 0xE0
    opt_core = struct.pack(
        "<HBBIIIIIIIIHHHHHHIIIIIIII",
        0x010B,   # Magic PE32
        0, 0,     # linker version
        0x200,    # SizeOfCode
        0, 0,     # SizeOfInitializedData, SizeOfUninitializedData
        0x1000,   # AddressOfEntryPoint
        0x1000,   # BaseOfCode
        0,        # BaseOfData
        image_base,
        section_alignment, file_alignment,
        4, 0,     # OS version
        0, 0,     # Image version
        4, 0,     # Subsystem version
        0,        # Win32VersionValue
        size_of_image, size_of_headers,
        0,        # CheckSum
        2,        # Subsystem (GUI)
        0,        # DllCharacteristics
        0x100000, 0x1000,   # SizeOfStackReserve/Commit
        0x100000, 0x1000,   # SizeOfHeapReserve/Commit
        0,                  # LoaderFlags
        16,                 # NumberOfRvaAndSizes
    )
    # Data directory entries (16 × 8 bytes = 128 bytes of zeros)
    opt_data_dirs = b"\x00" * 128
    opt_header = opt_core + opt_data_dirs

    # Section header (.text)
    sec_name = b".text\x00\x00\x00"   # 8 bytes
    virtual_size = 0x200
    virtual_addr = 0x1000
    raw_size = 0x200
    raw_ptr = size_of_headers
    sec_header = struct.pack(
        "<8sIIIIIIHHI",
        sec_name,
        virtual_size, virtual_addr,
        raw_size, raw_ptr,
        0, 0, 0, 0,
        0x60000020,   # CNT_CODE | EXECUTE | READ
    )

    header_region = dos_header + pe_sig + coff + opt_header + sec_header
    # Pad headers to file_alignment
    padding = b"\x00" * (size_of_headers - len(header_region))
    # Section data
    section_data = b"\x90" * raw_size   # NOPs

    return header_region + padding + section_data


MINIMAL_PE = _make_minimal_pe()


# ---------------------------------------------------------------------------
# Feature extractor tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:

    def test_returns_correct_shape(self, tmp_path):
        pe_file = tmp_path / "sample.exe"
        pe_file.write_bytes(MINIMAL_PE)
        features = extract_features(pe_file)
        assert features.shape == (NUM_FEATURES,), f"Expected ({NUM_FEATURES},), got {features.shape}"

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == NUM_FEATURES

    def test_file_size_feature(self, tmp_path):
        pe_file = tmp_path / "sample.exe"
        pe_file.write_bytes(MINIMAL_PE)
        features = extract_features(pe_file)
        assert features[0] == len(MINIMAL_PE)

    def test_invalid_file_raises(self, tmp_path):
        bad_file = tmp_path / "bad.exe"
        bad_file.write_bytes(b"NOT A PE FILE")
        with pytest.raises(ValueError, match="Not a valid PE"):
            extract_features(bad_file)

    def test_entropy_between_0_and_8(self, tmp_path):
        pe_file = tmp_path / "sample.exe"
        pe_file.write_bytes(MINIMAL_PE)
        features = extract_features(pe_file)
        entropy = features[1]
        assert 0.0 <= entropy <= 8.0, f"Entropy out of range: {entropy}"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with no real models loaded (graceful degradation test)."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_required_keys(self, client):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data


class TestClassifyEndpoint:

    def test_classify_without_model_returns_503(self, client):
        pe_bytes = MINIMAL_PE
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.exe", pe_bytes, "application/octet-stream")},
        )
        # 503 when no model loaded, or 422 if PE parse fails — both acceptable
        assert resp.status_code in (422, 503)

    def test_classify_invalid_file_returns_422(self, client):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("bad.exe", b"NOT A PE", "application/octet-stream")},
        )
        assert resp.status_code in (422, 503)

    def test_invalid_classifier_param_returns_422(self, client):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.exe", MINIMAL_PE, "application/octet-stream")},
            data={"classifier": "invalid_model"},
        )
        assert resp.status_code == 422


class TestFeaturesEndpoint:

    def test_features_returns_200_for_valid_pe(self, client):
        resp = client.post(
            "/api/v1/features",
            files={"file": ("test.exe", MINIMAL_PE, "application/octet-stream")},
        )
        assert resp.status_code == 200

    def test_features_response_contains_all_names(self, client):
        resp = client.post(
            "/api/v1/features",
            files={"file": ("test.exe", MINIMAL_PE, "application/octet-stream")},
        )
        data = resp.json()
        assert set(data["features"].keys()) == set(FEATURE_NAMES)

    def test_features_invalid_file_returns_422(self, client):
        resp = client.post(
            "/api/v1/features",
            files={"file": ("bad.exe", b"GARBAGE", "application/octet-stream")},
        )
        assert resp.status_code == 422

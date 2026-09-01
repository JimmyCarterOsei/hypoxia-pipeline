"""Shared offline fixtures.

Every fixture here is synthetic and written into a tmp cache, so the suite runs
with no network. That is a hard requirement, not a convenience: the pipeline's
whole claim is reproducibility, and a test suite that depends on NCBI being up
cannot demonstrate it.
"""

from __future__ import annotations

import gzip
import socket

import numpy as np
import pytest

from hypoxiapipe.ingest.cache import Cache
from hypoxiapipe.ingest.endpoints import EndpointSpec
from hypoxiapipe.ingest.spec import CohortSpec

#: A minimal GPL annotation exercising the cases that matter: two probes for one
#: gene (collapse), a retired symbol (CYR61 -> CCN1), a multi-target probe, and
#: a control probe with no symbol.
GPL_ANNOT = """^PLATFORM = GPL9999
!Platform_title = Test array
!platform_table_begin
ID\tGene symbol\tGene title
PROBE_1\tALDOA\taldolase A
PROBE_2\tALDOA\taldolase A second probe
PROBE_3\tCYR61\tcysteine rich 61
PROBE_4\tBNIP3 /// BNIP3P1\tambiguous target
PROBE_5\t\tcontrol probe
PROBE_6\tANLN\tanillin
PROBE_7\tESRP1\tepithelial splicing factor
PROBE_8\tSLC16A1\tmonocarboxylate transporter
!platform_table_end
"""

PROBES = [f"PROBE_{i}" for i in range(1, 9)]


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    """Fail any test that opens an outbound socket.

    The offline guarantee is enforced rather than asserted in a comment: if a
    fetcher ever loses its cache guard, the suite says so immediately instead
    of passing on a machine that happens to have network.
    """
    real_connect = socket.socket.connect

    def blocked(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in {"127.0.0.1", "::1", "localhost"}:
            return real_connect(self, address, *args, **kwargs)
        raise RuntimeError(f"test attempted a network connection to {address!r}")

    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.socket.connect = real_connect


def series_matrix(n_samples: int = 40) -> str:
    """Return a synthetic GEO series-matrix document with survival characteristics."""
    samples = [f"GSM{i:04d}" for i in range(1, n_samples + 1)]
    rng = np.random.default_rng(0)
    header = [
        '!Series_title\t"Test series"',
        '!Series_platform_id\t"GPL9999"',
        "!Sample_geo_accession\t" + "\t".join(f'"{s}"' for s in samples),
        "!Sample_characteristics_ch1\t"
        + "\t".join(f'"bcr free time: {12 + i * 2}"' for i in range(n_samples)),
        "!Sample_characteristics_ch1\t"
        + "\t".join(f'"bcr status: {1 if i % 3 == 0 else 0}"' for i in range(n_samples)),
        "!series_matrix_table_begin",
        "ID_REF\t" + "\t".join(samples),
    ]
    rows = [
        probe + "\t" + "\t".join(str(v) for v in rng.normal(8.0, 1.0, n_samples).round(4))
        for probe in PROBES
    ]
    return "\n".join([*header, *rows, "!series_matrix_table_end", ""])


@pytest.fixture
def primed_cache(tmp_path):
    """Return an offline cache pre-loaded with a series matrix and its platform."""
    cache = Cache(tmp_path / "cache", offline=True)
    cache.put(
        "geo/GSE00001_series_matrix.txt.gz",
        gzip.compress(series_matrix().encode()),
        url="fixture://series",
    )
    cache.put("geo/GPL9999.annot.gz", gzip.compress(GPL_ANNOT.encode()), url="fixture://gpl")
    return cache


@pytest.fixture
def geo_spec():
    """Return a cohort spec pointing at the fixture accession."""
    return CohortSpec(
        name="TestCohort",
        source="geo",
        accession="GSE00001",
        platform="GPL9999",
        endpoint=EndpointSpec(
            name="BCR",
            time_column="bcr_free_time",
            event_column="bcr_status",
            time_unit="months",
            cap_months=60,
        ),
    )

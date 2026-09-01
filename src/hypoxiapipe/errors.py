"""Exception hierarchy for hypoxiapipe.

Every failure mode this package cares about has its own class, because the
premise of the project is that data problems should fail loudly and specifically
rather than propagate into a plausible-looking hazard ratio. Catching
``HypoxiapipeError`` at the CLI boundary turns any of them into a non-zero exit
with a message that names the cause.
"""


class HypoxiapipeError(Exception):
    """Base class for all hypoxiapipe errors."""


# ---------------------------------------------------------------- signatures
class SignatureError(HypoxiapipeError):
    """Problem with a signature specification."""


class ChecksumMismatchError(SignatureError):
    """Signature gene set does not match its recorded checksum.

    Raised when a spec file's content hash disagrees with the stored
    checksum - i.e. the gene list was edited without re-verification.
    This is the provenance failure this package exists to prevent.
    """


class IncompleteSpecError(SignatureError):
    """Signature spec is missing required fields (e.g. empty gene list)."""


# ------------------------------------------------------------------- scoring
class ScoringError(HypoxiapipeError):
    """Problem during signature scoring."""


class InsufficientGenesError(ScoringError):
    """Too few signature genes present in the expression matrix."""


# -------------------------------------------------------------------- ingest
class IngestError(HypoxiapipeError):
    """Problem acquiring or assembling a cohort."""


class DownloadError(IngestError):
    """A remote resource could not be retrieved."""


class CacheMissError(IngestError):
    """Requested key is absent from the cache and offline mode is in force.

    Deliberately an error rather than a silent fetch: CI runs offline, so a
    missing fixture must fail the build instead of reaching for the network.
    """


class ParseError(IngestError):
    """A downloaded or supplied file could not be parsed as expected."""


class CohortAlignmentError(IngestError):
    """Expression matrix and clinical table do not describe the same samples."""


class EndpointError(IngestError):
    """Survival endpoint columns are missing, malformed, or unusable."""


# -------------------------------------------------------------- harmonisation
class HarmoniseError(HypoxiapipeError):
    """Problem mapping identifiers to a pinned symbol authority."""


class AliasTableError(HarmoniseError):
    """The alias table is missing, malformed, or fails its checksum."""


class AmbiguousSymbolError(HarmoniseError):
    """A symbol maps to more than one approved symbol under the pinned release."""


# ------------------------------------------------------------------------ QC
class QCFailedError(HypoxiapipeError):
    """A cohort failed one or more blocking QC checks."""


#: Readable synonym of :class:`QCFailedError`; both names catch the same class.
QCFailure = QCFailedError

"""Reader tests."""

import icechunk
import numpy as np
import obstore
import pytest
import xarray as xr

from titiler.multidim import reader


class _StopWiring(Exception):
    """Raised by stubs to stop execution once the wiring under test ran."""


@pytest.fixture
def ambient_aws(monkeypatch):
    """Ambient AWS credentials, from env vars so botocore makes no network call."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")


@pytest.fixture
def no_ambient_aws(monkeypatch, tmp_path):
    """No AWS credentials anywhere botocore looks (env, files, IMDS)."""
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _capture_s3store(monkeypatch):
    captured = {}

    def fake_s3store(**kwargs):
        captured.update(kwargs)
        raise _StopWiring

    monkeypatch.setattr(obstore.store, "S3Store", fake_s3store)
    with pytest.raises(_StopWiring):
        reader.identify_storage_backend("s3://some-random-bucket/f.nc")
    return captured


def _capture_s3_storage(monkeypatch):
    captured = {}

    def fake_s3_storage(**kwargs):
        captured.update(kwargs)
        raise _StopWiring

    monkeypatch.setattr(icechunk, "s3_storage", fake_s3_storage)
    with pytest.raises(_StopWiring):
        reader.opener_icechunk("s3://some-random-bucket/repo")
    return captured


def test_identify_uses_boto3_provider(monkeypatch, ambient_aws):
    """identify_storage_backend uses the ambient Boto3 provider when there is one."""
    from obstore.auth.boto3 import Boto3CredentialProvider

    captured = _capture_s3store(monkeypatch)
    assert isinstance(captured["credential_provider"], Boto3CredentialProvider)
    assert "skip_signature" not in captured


def test_identify_unsigned_without_credentials(monkeypatch, no_ambient_aws):
    """Without ambient credentials, public buckets are listed unsigned."""
    captured = _capture_s3store(monkeypatch)
    assert captured["skip_signature"] is True
    assert "credential_provider" not in captured


def test_opener_icechunk_uses_from_env_storage(monkeypatch, ambient_aws):
    """opener_icechunk opens s3 stores with from_env (ambient) credentials."""
    captured = _capture_s3_storage(monkeypatch)
    assert captured["from_env"] is True
    assert "anonymous" not in captured


def test_opener_icechunk_anonymous_without_credentials(monkeypatch, no_ambient_aws):
    """Without ambient credentials, icechunk s3 stores open anonymously."""
    captured = _capture_s3_storage(monkeypatch)
    assert captured["anonymous"] is True
    assert "from_env" not in captured


def _tiny_dataset():
    return xr.Dataset(
        {"data": (("y", "x"), np.ones((2, 2)))},
        coords={"x": [0, 1], "y": [1, 0]},
    ).rio.write_crs("EPSG:4326")


ENDPOINT = "https://archive.podaac.earthdata.nasa.gov/s3credentials"
REGISTRY_PREFIX = "s3://podaac-ops-cumulus-protected/MUR/"


class _StubRepo:
    def __init__(self, url_prefixes):
        class _Container:
            store = "s3"

            def __init__(self, url_prefix):
                self.url_prefix = url_prefix

        class _Config:
            virtual_chunk_containers = {
                f"c{i}": _Container(p) for i, p in enumerate(url_prefixes)
            }

        self.config = _Config()

    def readonly_session(self, branch):
        class _Session:
            store = object()

        return _Session()


def test_opener_icechunk_primes_only_declared_earthdata_containers(monkeypatch):
    """The opener primes EDL credentials only for earthdata containers the
    repo actually declares, and never touches EDL for a repo without
    earthdata containers."""
    primed = []
    monkeypatch.setattr(
        "titiler.multidim.earthdata.prime_earthdata_endpoints",
        lambda endpoints: primed.append(list(endpoints)),
    )
    monkeypatch.setattr(
        icechunk.Repository,
        "open",
        staticmethod(lambda **kwargs: _StubRepo([REGISTRY_PREFIX])),
    )
    monkeypatch.setattr(reader.xr, "open_dataset", lambda store, **k: _tiny_dataset())
    access = {
        REGISTRY_PREFIX: {"earthdata": True},
        "s3://nasa-waterinsight/NLDAS3/": {"anonymous": True},
    }
    reader.opener_icechunk("file:///tmp/repo", authorize_virtual_chunk_access=access)
    assert primed == [[ENDPOINT]]


def test_opener_icechunk_skips_earthdata_for_undeclared_containers(monkeypatch):
    """An earthdata entry for another repo's bucket must not couple this
    open to Earthdata availability or EULA state (a fully public dataset
    served alongside a protected one must never 403/500 on EDL problems)."""
    primed = []
    monkeypatch.setattr(
        "titiler.multidim.earthdata.prime_earthdata_endpoints",
        lambda endpoints: primed.append(list(endpoints)),
    )
    monkeypatch.setattr(
        icechunk.Repository,
        "open",
        staticmethod(lambda **kwargs: _StubRepo(["s3://nasa-waterinsight/NLDAS3/"])),
    )
    monkeypatch.setattr(reader.xr, "open_dataset", lambda store, **k: _tiny_dataset())
    access = {
        REGISTRY_PREFIX: {"earthdata": True},
        "s3://nasa-waterinsight/NLDAS3/": {"anonymous": True},
    }
    reader.opener_icechunk("file:///tmp/repo", authorize_virtual_chunk_access=access)
    assert primed == []

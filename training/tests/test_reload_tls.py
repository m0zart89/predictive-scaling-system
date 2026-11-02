"""reload_serving: the client certificate reaches urlopen for an https URL."""

import shutil
import subprocess

import pytest

from forecast_training import registry as reg

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not installed")


def test_reload_uses_a_client_certificate_when_a_tls_dir_is_given(tmp_path, monkeypatch):
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", tmp_path / "tls.key",
                    "-out", tmp_path / "tls.crt", "-days", "2", "-subj", "/CN=forecast-retrain"],
                   check=True, capture_output=True)
    shutil.copy(tmp_path / "tls.crt", tmp_path / "ca.crt")
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout, context=None):
        seen["url"], seen["context"], seen["token"] = req.full_url, context, req.get_header("X-reload-token")
        return Resp()

    monkeypatch.setattr(reg.urllib.request, "urlopen", fake_urlopen)
    assert reg.reload_serving("https://forecast-serving:8080", token="t", tls_dir=str(tmp_path)) is True
    assert seen["url"] == "https://forecast-serving:8080/reload" and seen["token"] == "t"
    assert seen["context"] is not None and seen["context"].verify_mode.name == "CERT_REQUIRED"
    assert reg.reload_serving("http://forecast-serving:8080") is True
    assert seen["context"] is None

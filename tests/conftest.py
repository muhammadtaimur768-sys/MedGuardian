import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def mock_fhir():
    from eval.mock_fhir_server import serve
    server = serve(port=8091, background=True)
    yield "http://127.0.0.1:8091/fhir/"
    server.shutdown()

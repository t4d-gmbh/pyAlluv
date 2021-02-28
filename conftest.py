import pytest

optional_markers = {
    "devtest": {
        "help": "run all tests including those for test-driven dev.",
        "description": "test is relevant for test-drive development only.",
        "skip-reason": "only ran during development, use --{marker} to run."
    }
}


def pytest_addoption(parser):
    for marker, info in optional_markers.items():
        parser.addoption(f"--{marker}", action="store_true", default=False,
                         help=info['help'])


def pytest_configure(config):
    for marker, info in optional_markers.items():
        config.addinivalue_line("markers", f"{marker}: {info['description']}")


def pytest_collection_modifyitems(config, items):
    for marker, info in optional_markers.items():
        if config.getoption(f"--{marker}"):
            return
        skip_devtest = pytest.mark.skip(
            reason=info['skip-reason'].format(marker=marker)
        )
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip_devtest)

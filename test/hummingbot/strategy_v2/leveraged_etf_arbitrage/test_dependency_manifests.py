import re
from pathlib import Path


EXCHANGE_CALENDARS_PIN = "exchange-calendars==4.13.2"


def _hummingbot_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").is_file() and (parent / "setup" / "environment.yml").is_file():
            return parent
    raise AssertionError("Hummingbot repository root not found from test path")


def test_supported_install_manifests_share_the_exact_exchange_calendars_pin():
    root = _hummingbot_root()
    manifests = (
        root / "setup.py",
        root / "setup" / "environment.yml",
        root / "setup" / "environment_dydx.yml",
        root / "setup" / "pip_packages.txt",
    )

    pin_counts = {
        manifest.relative_to(root): manifest.read_text().count(EXCHANGE_CALENDARS_PIN)
        for manifest in manifests
    }

    assert pin_counts == {manifest.relative_to(root): 1 for manifest in manifests}


def test_makefile_and_docker_install_paths_are_covered_by_the_pinned_manifests():
    root = _hummingbot_root()
    makefile = (root / "Makefile").read_text()
    dockerfile = (root / "Dockerfile").read_text()

    makefile_manifests = set(re.findall(r"setup/(?:environment(?:_dydx)?\.yml|pip_packages\.txt)", makefile))
    docker_manifests = set(re.findall(r"setup/(?:environment(?:_dydx)?\.yml|pip_packages\.txt)", dockerfile))

    assert makefile_manifests == {
        "setup/environment.yml",
        "setup/environment_dydx.yml",
        "setup/pip_packages.txt",
    }
    assert docker_manifests == {
        "setup/environment.yml",
        "setup/pip_packages.txt",
    }

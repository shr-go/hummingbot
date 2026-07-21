import ast
import re
from pathlib import Path

import yaml


EXCHANGE_CALENDARS_PIN = "exchange-calendars==4.13.2"


def _hummingbot_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").is_file() and (parent / "setup" / "environment.yml").is_file():
            return parent
    raise AssertionError("Hummingbot repository root not found from test path")


def _setup_install_requires(setup_path: Path) -> list[str]:
    setup_tree = ast.parse(setup_path.read_text())
    for node in ast.walk(setup_tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "install_requires" for target in node.targets)
            and isinstance(node.value, ast.List)
        ):
            return [value.value for value in node.value.elts if isinstance(value, ast.Constant)]
    raise AssertionError("setup.py install_requires list not found")


def _environment_pip_requirements(environment_path: Path) -> list[str]:
    dependencies = yaml.safe_load(environment_path.read_text())["dependencies"]
    pip_sections = [
        dependency["pip"]
        for dependency in dependencies
        if isinstance(dependency, dict) and "pip" in dependency
    ]
    assert len(pip_sections) == 1
    return pip_sections[0]


def _requirements_file_specs(requirements_path: Path) -> list[str]:
    return [
        line.partition("#")[0].strip()
        for line in requirements_path.read_text().splitlines()
        if line.partition("#")[0].strip()
    ]


def test_supported_install_manifests_share_the_exact_exchange_calendars_pin():
    root = _hummingbot_root()
    installation_requirements = {
        "setup.py install_requires": _setup_install_requires(root / "setup.py"),
        "default environment pip": _environment_pip_requirements(root / "setup" / "environment.yml"),
        "DYDX environment pip": _environment_pip_requirements(root / "setup" / "environment_dydx.yml"),
        "no-deps pip requirements": _requirements_file_specs(root / "setup" / "pip_packages.txt"),
    }

    pin_counts = {
        name: requirements.count(EXCHANGE_CALENDARS_PIN)
        for name, requirements in installation_requirements.items()
    }
    assert pin_counts == {
        name: 1 for name in installation_requirements
    }


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

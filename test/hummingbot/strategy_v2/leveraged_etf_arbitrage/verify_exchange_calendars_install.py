"""Fresh-install probe for the exact XNYS calendar dependency."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


PIN = "exchange-calendars==4.13.2"


def main() -> None:
    assert (3, 10, 12) <= sys.version_info < (4,), sys.version
    with tempfile.TemporaryDirectory(prefix="hummingbot-xnys-install-") as temporary_directory:
        target = Path(temporary_directory) / "site-packages"
        report_path = Path(temporary_directory) / "pip-report.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--quiet",
                "--target",
                str(target),
                "--report",
                str(report_path),
                PIN,
            ],
            check=True,
            timeout=120,
        )

        report = json.loads(report_path.read_text())
        resolved = {
            item["metadata"]["name"].lower().replace("_", "-"): item["metadata"]["version"]
            for item in report["install"]
        }
        assert resolved["exchange-calendars"] == "4.13.2", resolved

        probe = """
import importlib.metadata
import json
import pathlib
import sys

target = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import exchange_calendars

module_path = pathlib.Path(exchange_calendars.__file__).resolve()
assert module_path.is_relative_to(target), (module_path, target)
distribution = next(
    distribution
    for distribution in importlib.metadata.distributions(path=[target])
    if distribution.metadata["Name"].lower().replace("_", "-") == "exchange-calendars"
)
assert distribution.version == "4.13.2", distribution.version
print(json.dumps({"module": str(module_path), "version": distribution.version}, sort_keys=True))
"""
        completed_probe = subprocess.run(
            [sys.executable, "-I", "-c", probe, str(target)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(
            json.dumps(
                {
                    "fresh_import": json.loads(completed_probe.stdout),
                    "pin": PIN,
                    "python": sys.version.split()[0],
                    "resolved_distributions": resolved,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

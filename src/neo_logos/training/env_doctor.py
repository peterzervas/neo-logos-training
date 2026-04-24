"""Training environment checks for the supported Gemma 4 / RTX 5090 stack."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

from packaging.version import InvalidVersion, Version

SUPPORTED_PYTHON = (3, 12)
MIN_CUDA_VERSION = Version("12.8")
MIN_COMPUTE_CAPABILITY = (12, 0)

REQUIRED_MIN_VERSIONS = {
    "torch": "2.10.0",
    "triton": "3.3.1",
    "transformers": "5.0.0",
    "trl": "0.24.0",
    "unsloth": "2026.4.0",
    "unsloth_zoo": "2026.4.0",
    "bitsandbytes": "0.49.0",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    actual: str | None = None
    expected: str | None = None


def _installed_version(package: str) -> Version | None:
    try:
        return Version(metadata.version(package))
    except metadata.PackageNotFoundError:
        return None
    except InvalidVersion:
        return None


def _parse_cuda_version(value: str | None) -> Version | None:
    if not value:
        return None
    try:
        parts = value.split(".")
        if len(parts) < 2:
            return Version(value)
        return Version(f"{parts[0]}.{parts[1]}")
    except InvalidVersion:
        return None


def _status(failed: bool) -> str:
    return "FAIL" if failed else "PASS"


def _load_torch_module() -> Any:
    return __import__("torch")


def _package_checks() -> tuple[list[CheckResult], dict[str, str | None]]:
    checks = []
    versions = {}
    for package, minimum in REQUIRED_MIN_VERSIONS.items():
        installed = _installed_version(package)
        required = Version(minimum)
        versions[package] = str(installed) if installed else None
        if installed is None:
            checks.append(
                CheckResult(
                    name=f"package:{package}",
                    status="FAIL",
                    detail=f"{package} is not installed or has an invalid version",
                    actual=None,
                    expected=f">={required}",
                )
            )
        else:
            version_ok = installed >= required
            checks.append(
                CheckResult(
                    name=f"package:{package}",
                    status=_status(not version_ok),
                    detail=(
                        f"{package} {installed} >= {required}"
                        if version_ok
                        else f"{package} {installed} is below required {required}"
                    ),
                    actual=str(installed),
                    expected=f">={required}",
                )
            )
    return checks, versions


def _runtime_checks(require_cuda: bool) -> tuple[list[CheckResult], dict[str, Any]]:
    checks = []
    runtime: dict[str, Any] = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_cuda": None,
        "cuda_available": None,
        "gpu_name": None,
        "compute_capability": None,
    }

    python_actual = sys.version_info[:2]
    python_ok = python_actual == SUPPORTED_PYTHON
    checks.append(
        CheckResult(
            name="runtime:python",
            status=_status(not python_ok),
            detail=(
                f"Python {runtime['python']} matches supported "
                f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}"
                if python_ok
                else f"Python {runtime['python']} does not match supported "
                f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}"
            ),
            actual=runtime["python"],
            expected=f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}",
        )
    )

    try:
        torch = _load_torch_module()
    except Exception as exc:
        checks.append(
            CheckResult(
                name="runtime:torch_import",
                status="FAIL",
                detail=f"torch failed to import: {exc}",
            )
        )
        return checks, runtime

    torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    runtime["torch_cuda"] = torch_cuda
    parsed_cuda = _parse_cuda_version(torch_cuda)
    cuda_ok = parsed_cuda is not None and parsed_cuda >= MIN_CUDA_VERSION
    checks.append(
        CheckResult(
            name="runtime:torch_cuda",
            status=_status(not cuda_ok),
            detail=(
                f"torch CUDA runtime {torch_cuda} >= {MIN_CUDA_VERSION}"
                if cuda_ok
                else f"torch CUDA runtime {torch_cuda} is below required {MIN_CUDA_VERSION}"
            ),
            actual=torch_cuda,
            expected=f">={MIN_CUDA_VERSION}",
        )
    )

    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(cuda and cuda.is_available())
    runtime["cuda_available"] = cuda_available
    checks.append(
        CheckResult(
            name="runtime:cuda_available",
            status=_status(require_cuda and not cuda_available),
            detail=(
                "CUDA is available"
                if cuda_available
                else (
                    "CUDA is not available"
                    if require_cuda
                    else "CUDA is not available; allowed for non-GPU checks"
                )
            ),
            actual=str(cuda_available),
            expected="True" if require_cuda else "optional",
        )
    )

    if not cuda_available:
        return checks, runtime

    runtime["gpu_name"] = cuda.get_device_name(0)
    capability = cuda.get_device_capability(0)
    runtime["compute_capability"] = ".".join(str(part) for part in capability)
    capability_ok = tuple(capability) >= MIN_COMPUTE_CAPABILITY
    checks.append(
        CheckResult(
            name="runtime:compute_capability",
            status=_status(not capability_ok),
            detail=(
                f"GPU compute capability {runtime['compute_capability']} >= "
                f"{MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}"
                if capability_ok
                else f"GPU compute capability {runtime['compute_capability']} is below "
                f"{MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}"
            ),
            actual=runtime["compute_capability"],
            expected=f">={MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}",
        )
    )

    return checks, runtime


def collect_environment_report(require_cuda: bool = True) -> dict[str, Any]:
    """Return a structured report for the supported training environment."""
    package_results, versions = _package_checks()
    runtime_results, runtime = _runtime_checks(require_cuda=require_cuda)
    checks = package_results + runtime_results

    return {
        "supported_stack": {
            "python": f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}",
            "cuda": f">={MIN_CUDA_VERSION}",
            "compute_capability": (
                f">={MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}"
            ),
        },
        "versions": versions,
        "runtime": runtime,
        "checks": [asdict(check) for check in checks],
        "failures": [check.detail for check in checks if check.status == "FAIL"],
    }


def _format_report(report: dict[str, Any]) -> str:
    lines = ["Neo-Logos training environment doctor", ""]

    lines.append("Package versions:")
    for package, minimum in REQUIRED_MIN_VERSIONS.items():
        installed = report["versions"].get(package) or "missing"
        lines.append(f"  - {package}: {installed} (required >= {minimum})")

    runtime = report["runtime"]
    lines.extend(
        [
            "",
            "Runtime:",
            f"  - Python: {runtime['python']}",
            f"  - torch CUDA: {runtime['torch_cuda'] or 'unknown'}",
            f"  - CUDA available: {runtime['cuda_available']}",
            f"  - GPU: {runtime['gpu_name'] or 'not detected'}",
            f"  - Compute capability: {runtime['compute_capability'] or 'unknown'}",
            "",
            "Checks:",
        ]
    )

    for check in report["checks"]:
        prefix = "PASS" if check["status"] == "PASS" else "FAIL"
        lines.append(f"  - [{prefix}] {check['name']}: {check['detail']}")

    if report["failures"]:
        lines.append("")
        lines.append("Unsupported training environment:")
        lines.extend(f"  - {failure}" for failure in report["failures"])
    else:
        lines.append("")
        lines.append("Training environment check passed.")

    return "\n".join(lines)


def check_training_environment(
    logger=None,
    fatal: bool = True,
    require_cuda: bool = True,
) -> list[str]:
    """Check the supported 31B training path and return failure details."""
    report = collect_environment_report(require_cuda=require_cuda)
    issues = report["failures"]

    message = _format_report(report)
    if issues:
        if logger:
            logger.error(message)
        if fatal:
            raise RuntimeError(message)
    elif logger:
        logger.info(message)

    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the supported Neo-Logos Gemma 4 training stack"
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print failures but exit 0. Useful on non-GPU development machines.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured report as JSON.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Do not fail when CUDA is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = collect_environment_report(require_cuda=not args.no_cuda)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))

    if report["failures"] and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

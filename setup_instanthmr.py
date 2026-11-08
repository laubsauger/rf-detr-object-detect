"""Fetch the pinned upstream Python package and cache the selected ONNX graphs.

Run with the project venv after installing requirements-instanthmr.txt.
No Git operations or upstream demo/viewer dependencies are needed.
"""
import argparse
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

REVISION = "aa4b7df48e1a91aa3b61d8e7e7d7db1f60b6ef52"
WEIGHTS_REVISION = "1d26fa94fe4478c7cc5dc0f8ed6b12a2382fc5b4"
SOURCE = Path(__file__).resolve().parent / "vendor" / "InstantHMR"


def install_source():
    marker = SOURCE / "REVISION"
    if marker.exists() and marker.read_text().strip() == REVISION:
        return
    url = f"https://codeload.github.com/Naturalpad-studio/InstantHMR/zip/{REVISION}"
    print(f"Downloading InstantHMR {REVISION} ...", flush=True)
    with urlopen(url, timeout=120) as response:
        archive = ZipFile(io.BytesIO(response.read()))
    prefix = f"InstantHMR-{REVISION}/"
    for name in archive.namelist():
        relative = name.removeprefix(prefix)
        if name.startswith(prefix) and not name.endswith("/") and (
            relative.startswith("instanthmr/") or relative in ("LICENSE", "LICENSE-SAM", "README.md")
        ):
            path = SOURCE / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    marker.write_text(REVISION + "\n")


def validate_fit(fit):
    if type(fit) is not int or fit not in (0, 1, 2):
        raise ValueError("InstantHMR fit must be 0, 1 or 2")
    return fit


def resolve_assets(variant, fit=1):
    validate_fit(fit)
    from huggingface_hub import hf_hub_download

    def fetch(filename):
        return hf_hub_download("momolesang/InstantHMR", filename, revision=WEIGHTS_REVISION)

    suffix = f"-fit{fit}" if fit else ""
    return fetch(f"instanthmr-{variant}{suffix}.onnx"), fetch("mhr-body.onnx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medium-commercial",
                        choices=["nano", "tiny", "small", "medium", "medium-commercial"])
    parser.add_argument("--fit", type=int, choices=[0, 1, 2], default=1)
    args = parser.parse_args()
    install_source()
    model, body = resolve_assets(args.model, args.fit)
    print(f"Network: {model}")
    print(f"Body: {body}")
    print("Ready. Start python bridge.py and select an InstantHMR model in the dashboard.")


if __name__ == "__main__":
    main()

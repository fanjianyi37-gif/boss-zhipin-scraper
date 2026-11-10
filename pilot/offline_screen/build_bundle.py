#!/usr/bin/env python3
"""把最小离线包复制到一个目录。不复制仓库、密钥或 .git。"""

import hashlib
import os
import shutil
import sys

PACKAGE_FILES = (
    "runner.py",
    "pricing.py",
    "test_offline.py",
    "build_bundle.py",
    "README.md",
    "expectations.json",
    "preference.json",
    "public_fixture.json",
    "public_fixture_meta.json",
)


def package_dir():
    return os.path.dirname(os.path.abspath(__file__))


def repo_root():
    return os.path.abspath(os.path.join(package_dir(), "..", ".."))


def build_bundle(destination, adapter_source=None):
    destination = os.path.abspath(destination)
    os.makedirs(destination, exist_ok=True)
    source_dir = package_dir()
    for name in PACKAGE_FILES:
        shutil.copy2(os.path.join(source_dir, name), os.path.join(destination, name))
    adapter = adapter_source or os.path.join(repo_root(), "scripts", "deepseek_screen.py")
    shutil.copy2(adapter, os.path.join(destination, "deepseek_screen.py"))
    digest_lines = []
    for name in sorted(os.listdir(destination)):
        path = os.path.join(destination, name)
        if not os.path.isfile(path) or name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path_bytes(path)).hexdigest()
        digest_lines.append(f"{digest}  {name}")
    text = "\n".join(digest_lines) + "\n"
    with open(os.path.join(destination, "SHA256SUMS"), "w", encoding="utf-8") as handle:
        handle.write(text)
    return destination


def path_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: build_bundle.py DESTINATION", file=sys.stderr)
        return 2
    build_bundle(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

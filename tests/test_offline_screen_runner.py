"""构建隔离包并在包内跑离线测试。不读取密钥，不访问 DeepSeek。"""

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE = os.path.join(ROOT, "pilot", "offline_screen")
sys.path.insert(0, PACKAGE)

import build_bundle


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


class BundleTests(unittest.TestCase):
    def test_bundle_matches_adapter_and_passes_offline_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = os.path.join(tmp, "bundle")
            build_bundle.build_bundle(bundle)
            self.assertEqual(
                sha256(os.path.join(bundle, "deepseek_screen.py")),
                sha256(os.path.join(ROOT, "scripts", "deepseek_screen.py")),
            )
            names = set(os.listdir(bundle))
            self.assertIn("runner.py", names)
            self.assertIn("public_fixture.json", names)
            self.assertNotIn(".git", names)
            self.assertNotIn(".env", names)
            env = os.environ.copy()
            env.pop("DEEPSEEK_API_KEY", None)
            env["PYTHONPATH"] = bundle
            env["PYTHONSAFEPATH"] = "1"
            completed = subprocess.run(
                [sys.executable, "test_offline.py"],
                cwd=bundle,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("OK", completed.stderr)

            archive_path = os.path.join(tmp, "deepseek-screen-offline.zip")
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name in sorted(os.listdir(bundle)):
                    archive.write(os.path.join(bundle, name), arcname=name)
            with zipfile.ZipFile(archive_path) as archive:
                archived = set(archive.namelist())
            self.assertNotIn(".git", archived)
            self.assertNotIn(".env", archived)
            self.assertIn("deepseek_screen.py", archived)
            self.assertIn("README.md", archived)


if __name__ == "__main__":
    unittest.main()

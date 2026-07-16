from pathlib import Path


def test_core_never_imports_extensions():
    core = list(Path("src/kw").glob("*.py")) + list(Path("src/kw/learners").glob("*.py"))
    offenders = [p for p in core if "kw.extensions" in p.read_text()]
    assert offenders == []

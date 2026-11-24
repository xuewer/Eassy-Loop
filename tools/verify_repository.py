"""Check a source checkout without credentials, model calls, or run artifacts."""
import argparse
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

REQUIRED_FILES = (
    'README.md', '.env.example', '.gitignore', '.gitattributes', 'pytest.ini',
    'requirements.txt', 'requirements-dev.txt', 'requirements-lock.txt',
    '.github/workflows/tests.yml', 'src/__init__.py', 'src/__main__.py',
    'tests/conftest.py', 'data/examples.jsonl', 'data/exploration_v1.jsonl',
    'data/exploration_v1_manifest.json', 'docs/ARCHITECTURE.md',
    'docs/EXPERIMENTS.md', 'docs/DUAL_MODEL_V2.md', 'docs/REPRODUCIBILITY.md',
    'docs/TERMINAL_ENV.md', 'docs/ANNOTATION_GUIDE.md',
    'tools/Use-ProjectEnv.ps1', 'tools/verify_repository.py',
    'tools/verify_dual_results.py', 'tools/export_evidence.py',
)


def verify_repository(root: Path) -> dict:
    root = root.resolve()
    issues = [f'Missing required file: {name}' for name in REQUIRED_FILES
              if not (root / name).is_file()]
    python_files = sorted((root / 'src').glob('*.py'))
    python_files += sorted((root / 'tests').glob('*.py'))
    python_files += [root / name for name in REQUIRED_FILES
                     if name.startswith('tools/') and name.endswith('.py')]
    for path in python_files:
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            issues.append(f'Invalid Python: {path.relative_to(root)} ({type(exc).__name__})')
            continue
        for node in ast.walk(tree):
            modules = ([item.name for item in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                if not module.startswith('src.'):
                    continue
                target = root.joinpath(*module.split('.'))
                if not target.with_suffix('.py').is_file() and not (target / '__init__.py').is_file():
                    issues.append(f'Missing local module: {module} imported by {path.relative_to(root)}')
    documents = [root / 'README.md'] + [root / name for name in REQUIRED_FILES
                                        if name.startswith('docs/')]
    for path in documents:
        if not path.is_file():
            continue
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            link = link.split('#', 1)[0].split('?', 1)[0].strip('<>')
            if not link or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', link):
                continue
            target = (path.parent / unquote(link)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                issues.append(f'Broken local link: {path.relative_to(root)} -> {link}')
    if issues:
        raise ValueError('\n'.join(sorted(set(issues))))
    return {'status': 'passed', 'required_files': len(REQUIRED_FILES),
            'python_files_checked': len(python_files), 'documents_checked': len(documents),
            'model_calls': 0, 'requires_local_run_archive': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_repository(args.root)))
    except ValueError as exc:
        print(json.dumps({'status': 'failed', 'issues': str(exc).splitlines()}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

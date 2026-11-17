from pathlib import Path
import shutil

import pytest

from tools.verify_repository import REQUIRED_FILES, verify_repository


@pytest.fixture
def source_checkout(tmp_path):
    original = Path(__file__).resolve().parents[1]
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    for name in REQUIRED_FILES:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
    shutil.copytree(original / 'src', checkout / 'src', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    return checkout


def test_source_distribution_does_not_require_private_results(source_checkout):
    assert not (source_checkout / '.env').exists()
    assert not (source_checkout / 'runs').exists()
    assert not (source_checkout / 'experiments').exists()
    assert verify_repository(source_checkout)['status'] == 'passed'


def test_missing_transitive_module_is_reported(source_checkout):
    (source_checkout / 'src/transport.py').unlink()
    with pytest.raises(ValueError, match='src.transport'):
        verify_repository(source_checkout)


def test_removed_document_link_is_reported(source_checkout):
    with (source_checkout / 'README.md').open('a', encoding='utf-8') as handle:
        handle.write('\n[local document](docs/missing.md)\n')
    with pytest.raises(ValueError, match='Broken local link'):
        verify_repository(source_checkout)

"""
Test cross-platform file I/O and directory management

Tests for:
- Directory creation (Windows & Unix)
- JSON save/load operations
- Path handling with pathlib
- File listing and cleanup
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from configs.appconfig import USER_DIR

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_user_dir_creation():
    """Test that USER_DIR exists and can be created"""
    USER_DIR.mkdir(parents=True, exist_ok=True)
    assert USER_DIR.exists(), f'USER_DIR does not exist: {USER_DIR}'
    assert USER_DIR.is_dir(), f'USER_DIR is not a directory: {USER_DIR}'


def test_subdirectory_creation():
    """Test creating subdirectories (pygraphs, force_graphs)"""
    pygraphs_dir = USER_DIR / 'pygraphs'
    force_graphs_dir = USER_DIR / 'force_graphs'

    pygraphs_dir.mkdir(parents=True, exist_ok=True)
    force_graphs_dir.mkdir(parents=True, exist_ok=True)

    assert pygraphs_dir.exists(), f'pygraphs dir not created: {pygraphs_dir}'
    assert force_graphs_dir.exists(), f'force_graphs dir not created: {force_graphs_dir}'


def test_json_save_load():
    """Test saving and loading JSON files with pathlib"""
    test_dir = USER_DIR / 'test_files'
    test_dir.mkdir(parents=True, exist_ok=True)

    test_data = {
        'test': 'data',
        'numbers': [1, 2, 3],
        'nested': {'key': 'value'},
    }

    # Save JSON
    test_file = test_dir / 'test_data.json'
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    assert test_file.exists(), f'JSON file not created: {test_file}'

    # Load JSON
    with open(test_file) as f:
        loaded_data = json.load(f)

    assert loaded_data == test_data, "Loaded data doesn't match saved data"

    # Cleanup
    shutil.rmtree(test_dir)


def test_file_listing():
    """Test listing files with pathlib.glob()"""
    pygraphs_dir = USER_DIR / 'pygraphs'
    pygraphs_dir.mkdir(parents=True, exist_ok=True)

    # Create some test files
    for i in range(3):
        test_file = pygraphs_dir / f'test_graph_{i}.json'
        with open(test_file, 'w') as f:
            json.dump({'id': i}, f)

    # List all JSON files
    json_files = list(pygraphs_dir.glob('*.json'))
    assert len(json_files) >= 3, f'Expected at least 3 files, found {len(json_files)}'

    # Verify they are Path objects
    for file_path in json_files:
        assert isinstance(file_path, Path), f'Expected Path object, got {type(file_path)}'
        assert file_path.suffix == '.json', f'Expected .json suffix, got {file_path.suffix}'

    # Cleanup test files
    for test_file in pygraphs_dir.glob('test_graph_*.json'):
        test_file.unlink()


def test_path_operations():
    """Test cross-platform path operations"""
    test_path = USER_DIR / 'subdir' / 'file.json'

    # Test parts extraction
    assert test_path.name == 'file.json', f'Wrong name: {test_path.name}'
    assert test_path.stem == 'file', f'Wrong stem: {test_path.stem}'
    assert test_path.suffix == '.json', f'Wrong suffix: {test_path.suffix}'

    # Test parent navigation
    parent = test_path.parent
    assert parent.name == 'subdir', f'Wrong parent name: {parent.name}'


def test_timestamp_sorting():
    """Test sorting files by modification time"""
    test_dir = USER_DIR / 'timestamp_test'
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create files with delays
    files = []
    for i in range(3):
        test_file = test_dir / f'file_{i}.json'
        with open(test_file, 'w') as f:
            json.dump({'order': i}, f)
        files.append(test_file)
        time.sleep(0.01)  # Small delay to ensure different timestamps

    # Get most recent file
    most_recent = max(files, key=lambda f: f.stat().st_mtime)

    # Should be the last file created
    with open(most_recent) as f:
        data = json.load(f)

    assert data['order'] == 2, f"Expected order=2, got order={data['order']}"

    # Cleanup
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    print('\n' + '='*60)
    print('Testing File Operations (Cross-Platform)')
    print('='*60)

    test_user_dir_creation()
    test_user_dir_creation()
    test_subdirectory_creation()
    test_json_save_load()
    test_file_listing()
    test_path_operations()
    test_timestamp_sorting()
    print('✅ All file operation tests passed!')

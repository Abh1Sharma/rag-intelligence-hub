"""
Test configuration and basic setup
"""

import pytest
import os
from config import Config, config

def test_config_initialization():
    """Test that configuration initializes properly."""
    assert isinstance(config, Config)
    assert config.CHUNK_SIZE == 800
    assert config.CHUNK_OVERLAP == 100
    assert config.DEFAULT_TOP_K == 5

def test_directory_creation():
    """Test that required directories are created."""
    assert os.path.exists(config.UPLOAD_DIRECTORY)
    assert os.path.exists(config.TEMP_DIRECTORY)
    assert os.path.exists(config.CHROMA_PERSIST_DIRECTORY)
    assert os.path.exists(os.path.dirname(config.LOG_FILE))

def test_demo_document_exists():
    """Test that demo document exists."""
    assert os.path.exists(config.DEMO_DOCUMENT_PATH)
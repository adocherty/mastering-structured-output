import os
import pickle
import pytest
import tempfile
from pydantic_structure_definitions import (
    ArticleResponse1XML,
    DynamicPXUnpickler,
)


def test_backwards_compatible_unpickler():
    """Test that BackwardCompatibleUnpickler can properly load pickled ArticleResponse1XML data
    with and without search_mode parameter."""

    # Create an instance of the class instead of using the class itself
    article_instance = ArticleResponse1XML(
        title="Test Title", answer="This is a test answer", number=42
    )

    test_data = {"metadata": "test_metadata", "article": article_instance}

    # Create temporary file that will be automatically cleaned up
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Write test data to pickle file
        pickle.dump(test_data, temp_file)
        temp_file_path = temp_file.name

    try:
        # Test without search_mode
        with open(temp_file_path, "rb") as f:
            loaded_data = DynamicPXUnpickler(f).load()

        # Verify the loaded data matches original
        assert loaded_data["metadata"] == "test_metadata"
        assert isinstance(loaded_data["article"], ArticleResponse1XML)
        assert loaded_data["article"].title == "Test Title"
        assert loaded_data["article"].answer == "This is a test answer"
        assert loaded_data["article"].number == 42
        assert loaded_data["article"].__xml_search_mode__ == "strict"

        # Test with search_mode parameter
        with open(temp_file_path, "rb") as f:
            loaded_data = DynamicPXUnpickler(f, search_mode="unordered").load()

        print(loaded_data["article"].__xml_search_mode__)

        # Verify again with search_mode
        assert isinstance(loaded_data["article"], ArticleResponse1XML)
        assert loaded_data["article"].title == "Test Title"
        assert loaded_data["article"].answer == "This is a test answer"
        assert loaded_data["article"].number == 42
        assert loaded_data["article"].__xml_search_mode__ == "unordered"

    finally:
        # Cleanup - remove test file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

"""Tests for tagging functionality."""
import pytest

from src.core.tagging import LabelTagger, evaluate_tagging


def test_label_assignment():
    """Test label assignment with threshold."""
    tagger = LabelTagger()
    
    # Set labels
    tagger.set_labels("cuisine", ["Lebanese", "Italian", "Indian"])
    tagger.set_labels("diet", ["vegan", "spicy"])
    
    # Tag item
    text = "Delicious chicken shawarma with spicy sauce"
    results = tagger.assign_all_groups(text, top_n=2, threshold=0.3)
    
    assert "cuisine" in results
    assert "diet" in results


def test_tagging_threshold():
    """Test that threshold filters labels."""
    tagger = LabelTagger()
    tagger.set_labels("cuisine", ["Lebanese", "Chinese"])
    
    text = "Lebanese hummus plate"
    
    # Low threshold
    results_low = tagger.assign_labels(text, "cuisine", top_n=5, threshold=0.1)
    
    # High threshold
    results_high = tagger.assign_labels(text, "cuisine", top_n=5, threshold=0.7)
    
    # High threshold should return fewer or equal labels
    assert len(results_high) <= len(results_low)


def test_tagging_evaluation():
    """Test tagging evaluation metrics."""
    predictions = [
        ["Lebanese", "Seafood"],
        ["Italian"],
        ["Indian", "Spicy"],
    ]
    
    ground_truth = [
        ["Lebanese", "Arabic"],
        ["Italian", "Pizza"],
        ["Indian"],
    ]
    
    metrics = evaluate_tagging(predictions, ground_truth)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["f1"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

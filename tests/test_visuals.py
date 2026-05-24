import os
import pytest
from cantuccio import visuals

def test_colorlists():
    lists_to_test = [
        visuals.DEFAULT_COLORLIST,
        visuals.CATEGORICAL_COLORLIST,
        visuals.SEQUENTIAL_COLORLIST,
        visuals.SEQUENTIAL_COLORLIST_v2,
        visuals.DIVERGING_COLORLIST,
        visuals.DIVERGING_COLORLIST_v2,
    ]
    for color_list in lists_to_test:
        assert isinstance(color_list, list)
        assert len(color_list) > 0
        for color in color_list:
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7

def test_get_paper_style_success():
    style = visuals.get_paper_style(journal="prd", cols="onecol", aspect=1.5)
    
    assert isinstance(style, list)
    assert len(style) == 2
    
    style_path, rc_params = style
    assert isinstance(style_path, str)
    assert style_path.endswith("paper.mplstyle")
    assert os.path.exists(style_path)
    
    assert isinstance(rc_params, dict)
    assert "figure.figsize" in rc_params
    
    width, height = rc_params["figure.figsize"]
    assert width / height == pytest.approx(1.5)
    
    # Test CQG journal with default golden ratio
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    style_cqg = visuals.get_paper_style(journal="cqg", cols="onecol")
    w, h = style_cqg[1]["figure.figsize"]
    assert w / h == pytest.approx(golden_ratio)

def test_get_paper_style_invalid_journal():
    with pytest.raises(ValueError, match="Journal 'invalid_journal' not found"):
        visuals.get_paper_style(journal="invalid_journal")

def test_get_paper_style_invalid_cols():
    with pytest.raises(ValueError, match="Column option 'invalid_col' not found for journal 'prd'"):
        visuals.get_paper_style(journal="prd", cols="invalid_col")

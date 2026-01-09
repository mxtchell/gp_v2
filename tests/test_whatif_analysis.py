from whatif_analysis import whatif_analysis
from skill_framework import SkillInput
from skill_framework.preview import preview_skill


class TestWhatIfAnalysis:
    """Test the what-if analysis skill"""

    preview = True

    def _run_whatif(self, parameters: dict, preview: bool = None):
        if preview is None:
            preview = self.preview
        skill_input: SkillInput = whatif_analysis.create_input(arguments=parameters)
        out = whatif_analysis(skill_input)
        if preview:
            preview_skill(whatif_analysis, out)
        return out

    def test_basic_whatif_europe_25pct(self):
        """Test what-if with 25% claims increase in Europe by country"""
        parameters = {
            "impact_pct": 25,
            "breakouts": ["country"],
            "periods": ["q2 2025"],
            "other_filters": [{"dim": "geo", "op": "=", "val": "europe"}]
        }
        self._run_whatif(parameters)

    def test_whatif_10pct_no_filter(self):
        """Test what-if with 10% claims increase, no region filter"""
        parameters = {
            "impact_pct": 10,
            "breakouts": ["country"],
            "periods": ["2024"]
        }
        self._run_whatif(parameters)

    def test_whatif_negative_impact(self):
        """Test what-if with claims decrease (-15%)"""
        parameters = {
            "impact_pct": -15,
            "breakouts": ["country"],
            "periods": ["q2 2025"],
            "other_filters": [{"dim": "geo", "op": "=", "val": "europe"}]
        }
        self._run_whatif(parameters)


if __name__ == "__main__":
    test = TestWhatIfAnalysis()
    test.test_basic_whatif_europe_25pct()

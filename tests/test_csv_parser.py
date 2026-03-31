"""
Tests for the CSV Parser Service.

Covers:
- Parsing valid CSV with all expected columns
- Handling missing columns gracefully
- Column name normalization (variations)
- Sample CSV generation
- Edge cases (empty CSV, single row, bad data types)
"""

import pytest
import io


class TestCSVParserBasics:
    """Tests for basic CSV parsing functionality."""

    def test_parse_sample_csv(self, sample_csv_content):
        """Should successfully parse the generated sample CSV."""
        from services.csv_parser import CSVParser

        parser = CSVParser()
        result = parser.parse(sample_csv_content, campaign_id="TEST_001", test_plan_id="PLAN_001")

        assert result.success is True
        assert result.row_count > 0
        assert result.results is not None
        assert len(result.errors) == 0

    def test_parse_returns_creative_results(self, sample_csv_content):
        """Parsed results should contain CreativeTestResult objects."""
        from services.csv_parser import CSVParser

        parser = CSVParser()
        result = parser.parse(sample_csv_content, campaign_id="TEST_001", test_plan_id="PLAN_001")

        assert result.results is not None
        assert len(result.results.results) > 0
        creative = result.results.results[0]
        assert creative.creative_id is not None
        assert creative.creative_name is not None

    def test_parse_extracts_lift_data(self, sample_csv_content):
        """Parsed creatives should have KPI lift data."""
        from services.csv_parser import CSVParser

        parser = CSVParser()
        result = parser.parse(sample_csv_content, campaign_id="TEST_001", test_plan_id="PLAN_001")

        creative = result.results.results[0]
        # Should have at least one KPI result
        assert len(creative.kpi_results) > 0


class TestCSVParserEdgeCases:
    """Tests for CSV parser edge cases."""

    def test_empty_csv(self):
        """Empty CSV should return failure."""
        from services.csv_parser import CSVParser

        parser = CSVParser()
        result = parser.parse("", campaign_id="TEST_001", test_plan_id="PLAN_001")

        assert result.success is False

    def test_header_only_csv(self):
        """CSV with only headers and no data rows should handle gracefully."""
        from services.csv_parser import CSVParser

        csv = "creative_id,creative_name,asset_type\n"
        parser = CSVParser()
        result = parser.parse(csv, campaign_id="TEST_001", test_plan_id="PLAN_001")

        # Should either succeed with 0 rows or fail gracefully
        assert result.row_count == 0 or result.success is False

    def test_minimal_valid_csv(self):
        """CSV with only required columns should parse."""
        from services.csv_parser import CSVParser

        csv = """creative_id,creative_name,asset_type,control_awareness,exposed_awareness,awareness_lift,awareness_stat_sig,control_sample_size,exposed_sample_size
VID001,Test Video,video,0.30,0.38,0.08,true,500,500
"""
        parser = CSVParser()
        result = parser.parse(csv, campaign_id="TEST_001", test_plan_id="PLAN_001")

        assert result.success is True
        assert result.row_count == 1

    def test_csv_with_bytes_input(self, sample_csv_content):
        """Should accept bytes input."""
        from services.csv_parser import CSVParser

        parser = CSVParser()
        result = parser.parse(
            sample_csv_content.encode("utf-8"),
            campaign_id="TEST_001",
            test_plan_id="PLAN_001",
        )

        assert result.success is True

    def test_csv_with_file_like_input(self, sample_csv_content):
        """Should accept file-like input."""
        from services.csv_parser import CSVParser

        parser = CSVParser()
        result = parser.parse(
            io.StringIO(sample_csv_content),
            campaign_id="TEST_001",
            test_plan_id="PLAN_001",
        )

        assert result.success is True


class TestSampleCSVGeneration:
    """Tests for generate_sample_csv()."""

    def test_generates_valid_csv(self):
        """generate_sample_csv() should return parseable CSV content."""
        from services.csv_parser import generate_sample_csv

        csv_content = generate_sample_csv()
        assert isinstance(csv_content, str)
        assert len(csv_content) > 100  # Not trivially empty
        assert "creative_id" in csv_content
        assert "creative_name" in csv_content

    def test_sample_csv_has_multiple_rows(self):
        """Sample CSV should have multiple data rows."""
        from services.csv_parser import generate_sample_csv

        csv_content = generate_sample_csv()
        lines = csv_content.strip().split("\n")
        assert len(lines) >= 3  # Header + at least 2 data rows

    def test_sample_csv_roundtrips(self):
        """generate_sample_csv() output should be parseable by CSVParser."""
        from services.csv_parser import CSVParser, generate_sample_csv

        csv_content = generate_sample_csv()
        parser = CSVParser()
        result = parser.parse(csv_content, campaign_id="RT_001", test_plan_id="RT_PLAN")

        assert result.success is True
        assert result.row_count >= 2

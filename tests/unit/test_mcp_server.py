from __future__ import annotations

from datetime import datetime, timezone

from mem_zero.mcp_server import _format_record


class TestFormatRecord:
    def test_converts_epoch_to_iso(self) -> None:
        record = {
            "id": "abc",
            "text": "hello",
            "created_at": 1779494400.0,
            "updated_at": 1779494400.0,
        }
        result = _format_record(record)
        expected = datetime.fromtimestamp(1779494400.0, tz=timezone.utc).isoformat()
        assert result["created_at"] == expected
        assert result["updated_at"] == expected

    def test_preserves_other_fields(self) -> None:
        record = {
            "id": "abc",
            "text": "hello",
            "created_at": 1779494400.0,
            "updated_at": 1779494400.0,
            "user_id": "john",
            "score": 0.95,
        }
        result = _format_record(record)
        assert result["id"] == "abc"
        assert result["text"] == "hello"
        assert result["user_id"] == "john"
        assert result["score"] == 0.95

    def test_handles_zero_timestamp(self) -> None:
        record = {"created_at": 0, "updated_at": 0}
        result = _format_record(record)
        assert result["created_at"] == 0
        assert result["updated_at"] == 0

    def test_handles_missing_timestamps(self) -> None:
        record = {"id": "abc", "text": "hello"}
        result = _format_record(record)
        assert "created_at" not in result

    def test_output_is_parseable_iso(self) -> None:
        ts = datetime(2026, 5, 20, 12, 30, 0, tzinfo=timezone.utc).timestamp()
        record = {"created_at": ts, "updated_at": ts}
        result = _format_record(record)
        parsed = datetime.fromisoformat(result["created_at"])
        assert parsed.year == 2026
        assert parsed.month == 5
        assert parsed.day == 20
        assert parsed.hour == 12
        assert parsed.tzinfo == timezone.utc

    def test_handles_int_timestamps(self) -> None:
        record = {"created_at": 1779494400, "updated_at": 1779494400}
        result = _format_record(record)
        assert "2026" in result["created_at"]

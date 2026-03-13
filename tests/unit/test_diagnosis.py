"""Tests for the composite network diagnosis tool.

Tests the pure helper functions (_classify_device_type, _summarize_devices,
_slim_event) and the async network_diagnosis tool function.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("UNIFI_HOST", "127.0.0.1")
os.environ.setdefault("UNIFI_USERNAME", "test")
os.environ.setdefault("UNIFI_PASSWORD", "test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_helpers():
    from src.tools.diagnosis import _classify_device_type, _slim_event, _summarize_devices

    return _classify_device_type, _slim_event, _summarize_devices


def _import_network_diagnosis():
    from src.tools.diagnosis import network_diagnosis

    return network_diagnosis


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DEVICES = [
    {
        "type": "uap",
        "name": "Living Room AP",
        "mac": "aa:bb:cc:00:00:01",
        "model": "U6-Pro",
        "ip": "10.0.1.10",
        "state": 1,
    },
    {
        "type": "usw",
        "name": "Core Switch",
        "mac": "aa:bb:cc:00:00:02",
        "model": "USW-24",
        "ip": "10.0.1.11",
        "state": 1,
    },
    {"type": "ugw", "name": "Gateway", "mac": "aa:bb:cc:00:00:03", "model": "UDM-Pro", "ip": "10.0.1.1", "state": 1},
    {"type": "uap", "name": "Garage AP", "mac": "aa:bb:cc:00:00:04", "model": "U6-Lite", "ip": "10.0.1.12", "state": 0},
    {"type": "usp", "name": "PDU", "mac": "aa:bb:cc:00:00:05", "model": "USP-PDU-Pro", "ip": "10.0.1.13", "state": 5},
]


# ---------------------------------------------------------------------------
# TestClassifyDeviceType
# ---------------------------------------------------------------------------


class TestClassifyDeviceType:
    """Pure function: prefix-based device classification."""

    @pytest.mark.parametrize(
        "type_str,expected",
        [
            ("uap-ac-pro", "ap"),
            ("usw-24-poe", "switch"),
            ("usk-8", "switch"),
            ("ugw-3", "gateway"),
            ("udm-pro", "gateway"),
            ("uxg-pro", "gateway"),
            ("usp-pdu", "pdu"),
        ],
    )
    def test_known_prefixes(self, type_str, expected):
        classify, _, _ = _import_helpers()
        assert classify(type_str) == expected

    def test_unknown_prefix(self):
        classify, _, _ = _import_helpers()
        assert classify("xyz123") == "other"

    def test_empty_string(self):
        classify, _, _ = _import_helpers()
        assert classify("") == "other"


# ---------------------------------------------------------------------------
# TestSummarizeDevices
# ---------------------------------------------------------------------------


class TestSummarizeDevices:
    """Pure function: token-efficient device summary builder."""

    @staticmethod
    def _fn():
        _, _, summarize = _import_helpers()
        return summarize

    def test_mixed_types(self):
        result = self._fn()(SAMPLE_DEVICES, include_device_list=False)
        assert result["by_type"] == {"ap": 2, "switch": 1, "gateway": 1, "pdu": 1}

    def test_mixed_states(self):
        result = self._fn()(SAMPLE_DEVICES, include_device_list=False)
        assert result["by_status"]["online"] == 3
        assert result["by_status"]["offline"] == 1
        assert result["by_status"]["provisioning"] == 1

    def test_offline_devices_collected(self):
        result = self._fn()(SAMPLE_DEVICES, include_device_list=False)
        names = [d["name"] for d in result["offline_devices"]]
        assert "Garage AP" in names
        assert "PDU" in names
        assert len(result["offline_devices"]) == 2

    def test_all_online(self):
        online = [d | {"state": 1} for d in SAMPLE_DEVICES]
        result = self._fn()(online, include_device_list=False)
        assert result["offline_devices"] == []

    def test_include_device_list_false(self):
        result = self._fn()(SAMPLE_DEVICES, include_device_list=False)
        assert "devices" not in result

    def test_include_device_list_true(self):
        result = self._fn()(SAMPLE_DEVICES, include_device_list=True)
        assert "devices" in result
        assert len(result["devices"]) == len(SAMPLE_DEVICES)
        device = result["devices"][0]
        assert set(device.keys()) == {"name", "mac", "model", "type", "ip", "status"}

    def test_empty_input(self):
        result = self._fn()([], include_device_list=False)
        assert result["total"] == 0
        assert result["by_type"] == {}
        assert result["by_status"] == {}
        assert result["offline_devices"] == []

    def test_total_count(self):
        result = self._fn()(SAMPLE_DEVICES, include_device_list=False)
        assert result["total"] == 5


# ---------------------------------------------------------------------------
# TestSlimEvent
# ---------------------------------------------------------------------------


class TestSlimEvent:
    """Pure function: event field extraction with fallbacks."""

    @staticmethod
    def _fn():
        _, slim, _ = _import_helpers()
        return slim

    def test_primary_fields(self):
        event = {
            "time": 1700000000,
            "key": "EVT_AP_Lost_Contact",
            "msg": "AP lost",
            "mac": "aa:bb:cc:dd:ee:ff",
            "name": "Office AP",
        }
        result = self._fn()(event)
        assert result == {
            "time": 1700000000,
            "type": "EVT_AP_Lost_Contact",
            "message": "AP lost",
            "mac": "aa:bb:cc:dd:ee:ff",
            "name": "Office AP",
        }

    def test_fallback_fields(self):
        event = {
            "datetime": "2024-01-01T00:00:00Z",
            "subsystem": "wlan",
            "message": "client joined",
            "client": "11:22:33:44:55:66",
            "hostname": "laptop",
        }
        result = self._fn()(event)
        assert result == {
            "time": "2024-01-01T00:00:00Z",
            "type": "wlan",
            "message": "client joined",
            "mac": "11:22:33:44:55:66",
            "name": "laptop",
        }

    def test_missing_fields(self):
        result = self._fn()({})
        assert result["time"] is None
        assert result["type"] is None
        assert result["message"] == ""
        assert result["mac"] is None
        assert result["name"] is None


# ---------------------------------------------------------------------------
# TestNetworkDiagnosis
# ---------------------------------------------------------------------------


class TestNetworkDiagnosis:
    """Async tool: composite network diagnosis with graceful degradation."""

    @pytest.fixture
    def mock_managers(self):
        with (
            patch("src.tools.diagnosis.system_manager") as sys_mgr,
            patch("src.tools.diagnosis.device_manager") as dev_mgr,
            patch("src.tools.diagnosis._get_event_manager") as get_evt,
        ):
            evt_mgr = MagicMock()
            get_evt.return_value = evt_mgr

            sys_mgr.get_system_info = AsyncMock(return_value={"version": "8.6.9"})
            sys_mgr.get_network_health = AsyncMock(return_value=[{"subsystem": "wan", "status": "ok"}])
            evt_mgr.get_alarms = AsyncMock(return_value=[])
            evt_mgr.get_events = AsyncMock(return_value=[])
            dev_mgr.get_devices = AsyncMock(return_value=[])

            yield sys_mgr, evt_mgr, dev_mgr

    @pytest.mark.asyncio
    async def test_all_succeed(self, mock_managers):
        result = await _import_network_diagnosis()()

        assert result["success"] is True
        assert result["system_info"] == {"version": "8.6.9"}
        assert result["network_health"] == [{"subsystem": "wan", "status": "ok"}]
        assert result["alarms"]["count"] == 0
        assert result["events"]["count"] == 0
        assert result["device_summary"]["total"] == 0

    @pytest.mark.asyncio
    async def test_alarms_not_truncated(self, mock_managers):
        _, evt_mgr, _ = mock_managers
        evt_mgr.get_alarms = AsyncMock(return_value=[{"id": i} for i in range(5)])

        result = await _import_network_diagnosis()()

        assert result["alarms"]["count"] == 5
        assert result["alarms"]["truncated"] is False
        assert len(result["alarms"]["items"]) == 5

    @pytest.mark.asyncio
    async def test_alarms_truncated(self, mock_managers):
        _, evt_mgr, _ = mock_managers
        evt_mgr.get_alarms = AsyncMock(return_value=[{"id": i} for i in range(15)])

        result = await _import_network_diagnosis()()

        assert result["alarms"]["count"] == 15
        assert result["alarms"]["truncated"] is True
        assert len(result["alarms"]["items"]) == 10

    @pytest.mark.asyncio
    async def test_events_truncated_with_slim(self, mock_managers):
        _, evt_mgr, _ = mock_managers
        raw_events = [{"time": i, "key": f"EVT_{i}", "msg": f"msg {i}"} for i in range(12)]
        evt_mgr.get_events = AsyncMock(return_value=raw_events)

        result = await _import_network_diagnosis()()

        assert result["events"]["count"] == 12
        assert result["events"]["truncated"] is True
        assert len(result["events"]["items"]) == 10
        # Verify _slim_event was applied
        assert set(result["events"]["items"][0].keys()) == {"time", "type", "message", "mac", "name"}

    @pytest.mark.asyncio
    async def test_system_info_failure(self, mock_managers):
        sys_mgr, _, _ = mock_managers
        sys_mgr.get_system_info = AsyncMock(side_effect=RuntimeError("connection refused"))

        result = await _import_network_diagnosis()()

        assert result["success"] is True
        assert "error" in result["system_info"]
        assert "connection refused" in result["system_info"]["error"]
        # Other sections still populated
        assert result["network_health"] == [{"subsystem": "wan", "status": "ok"}]
        assert result["device_summary"]["total"] == 0

    @pytest.mark.asyncio
    async def test_device_failure(self, mock_managers):
        _, _, dev_mgr = mock_managers
        dev_mgr.get_devices = AsyncMock(side_effect=RuntimeError("timeout"))

        result = await _import_network_diagnosis()()

        assert result["success"] is True
        assert "error" in result["device_summary"]
        assert result["device_summary"]["total"] == 0
        # Other sections still populated
        assert result["system_info"] == {"version": "8.6.9"}

    @pytest.mark.asyncio
    async def test_all_fail(self, mock_managers):
        sys_mgr, evt_mgr, dev_mgr = mock_managers
        sys_mgr.get_system_info = AsyncMock(side_effect=RuntimeError("fail1"))
        sys_mgr.get_network_health = AsyncMock(side_effect=RuntimeError("fail2"))
        evt_mgr.get_alarms = AsyncMock(side_effect=RuntimeError("fail3"))
        evt_mgr.get_events = AsyncMock(side_effect=RuntimeError("fail4"))
        dev_mgr.get_devices = AsyncMock(side_effect=RuntimeError("fail5"))

        result = await _import_network_diagnosis()()

        assert result["success"] is True
        assert "error" in result["system_info"]
        assert "error" in result["network_health"]
        assert "error" in result["alarms"]
        assert "error" in result["events"]
        assert "error" in result["device_summary"]

    @pytest.mark.asyncio
    async def test_include_device_list_true(self, mock_managers):
        _, _, dev_mgr = mock_managers
        dev_mgr.get_devices = AsyncMock(
            return_value=[
                {
                    "type": "uap",
                    "name": "AP1",
                    "mac": "aa:bb:cc:00:00:01",
                    "model": "U6",
                    "ip": "10.0.1.10",
                    "state": 1,
                },
            ]
        )

        result = await _import_network_diagnosis()(include_device_list=True)

        assert "devices" in result["device_summary"]
        assert len(result["device_summary"]["devices"]) == 1

    @pytest.mark.asyncio
    async def test_include_device_list_false_default(self, mock_managers):
        _, _, dev_mgr = mock_managers
        dev_mgr.get_devices = AsyncMock(
            return_value=[
                {
                    "type": "uap",
                    "name": "AP1",
                    "mac": "aa:bb:cc:00:00:01",
                    "model": "U6",
                    "ip": "10.0.1.10",
                    "state": 1,
                },
            ]
        )

        result = await _import_network_diagnosis()()

        assert "devices" not in result["device_summary"]

    @pytest.mark.asyncio
    async def test_devices_with_raw_attribute(self, mock_managers):
        """aiounifi model objects have a .raw dict attribute."""
        _, _, dev_mgr = mock_managers
        model_obj = MagicMock()
        model_obj.raw = {
            "type": "usw",
            "name": "Switch",
            "mac": "aa:bb:cc:00:00:02",
            "model": "USW-24",
            "ip": "10.0.1.11",
            "state": 1,
        }
        dev_mgr.get_devices = AsyncMock(return_value=[model_obj])

        result = await _import_network_diagnosis()()

        assert result["device_summary"]["total"] == 1
        assert result["device_summary"]["by_type"] == {"switch": 1}

    @pytest.mark.asyncio
    async def test_devices_as_raw_dicts(self, mock_managers):
        """Plain dicts pass through without .raw extraction."""
        _, _, dev_mgr = mock_managers
        dev_mgr.get_devices = AsyncMock(
            return_value=[
                {"type": "ugw", "name": "GW", "mac": "aa:bb:cc:00:00:03", "model": "UDM", "ip": "10.0.1.1", "state": 1},
            ]
        )

        result = await _import_network_diagnosis()()

        assert result["device_summary"]["by_type"] == {"gateway": 1}

    @pytest.mark.asyncio
    async def test_event_manager_init_failure(self, mock_managers):
        """_get_event_manager() raises — alarms/events degrade, others succeed."""
        with patch("src.tools.diagnosis._get_event_manager", side_effect=RuntimeError("import failed")):
            result = await _import_network_diagnosis()()

        assert result["success"] is True
        # Alarms and events degrade to error dicts
        assert "error" in result["alarms"]
        assert "import failed" in result["alarms"]["error"]
        assert "error" in result["events"]
        assert "import failed" in result["events"]["error"]
        # Other sections still populated
        assert result["system_info"] == {"version": "8.6.9"}
        assert result["network_health"] == [{"subsystem": "wan", "status": "ok"}]
        assert result["device_summary"]["total"] == 0

"""
UniFi Network MCP composite diagnosis tool.

Fetches system info, network health, alarms, events, and device summary
in a single parallel call, returning a token-efficient snapshot for
diagnostic triage.
"""

import asyncio
import logging
from typing import Any, Dict

from src.runtime import device_manager, server, system_manager

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies (same pattern as events.py)
_event_manager = None


def _get_event_manager():
    """Lazy-load the event manager to avoid circular imports."""
    global _event_manager
    if _event_manager is None:
        from src.managers.event_manager import EventManager
        from src.runtime import get_connection_manager

        _event_manager = EventManager(get_connection_manager())
    return _event_manager


# Device type prefix → human-readable category
_DEVICE_TYPE_MAP = {
    "uap": "ap",
    "usw": "switch",
    "usk": "switch",
    "ugw": "gateway",
    "udm": "gateway",
    "uxg": "gateway",
    "usp": "pdu",
}

# Device state code → human-readable status
_STATE_MAP = {
    0: "offline",
    1: "online",
    2: "pending",
    4: "adopting",
    5: "provisioning",
    6: "upgrading",
    11: "heartbeat_missed",
}


def _classify_device_type(type_str: str) -> str:
    """Map a UniFi device type string to a summary category."""
    for prefix, category in _DEVICE_TYPE_MAP.items():
        if type_str.startswith(prefix):
            return category
    return "other"


def _summarize_devices(devices_raw: list, include_device_list: bool) -> Dict[str, Any]:
    """Build a token-efficient device summary from raw device data."""
    by_type: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    offline_devices = []

    for d in devices_raw:
        # Count by type
        dtype = _classify_device_type(d.get("type", ""))
        by_type[dtype] = by_type.get(dtype, 0) + 1

        # Count by status
        state = d.get("state", 0)
        status_str = _STATE_MAP.get(state, "unknown")
        by_status[status_str] = by_status.get(status_str, 0) + 1

        # Collect offline devices
        if state != 1:
            offline_devices.append(
                {
                    "name": d.get("name", d.get("model", "Unknown")),
                    "mac": d.get("mac", ""),
                    "model": d.get("model", ""),
                    "status": status_str,
                }
            )

    result: Dict[str, Any] = {
        "total": len(devices_raw),
        "by_type": by_type,
        "by_status": by_status,
        "offline_devices": offline_devices,
    }

    if include_device_list:
        result["devices"] = [
            {
                "name": d.get("name", d.get("model", "Unknown")),
                "mac": d.get("mac", ""),
                "model": d.get("model", ""),
                "type": d.get("type", ""),
                "ip": d.get("ip", ""),
                "status": _STATE_MAP.get(d.get("state", 0), "unknown"),
            }
            for d in devices_raw
        ]

    return result


def _slim_event(event: dict) -> dict:
    """Extract only the diagnostic-relevant fields from an event."""
    return {
        "time": event.get("time") or event.get("datetime"),
        "type": event.get("key") or event.get("subsystem"),
        "message": event.get("msg") or event.get("message", ""),
        "mac": event.get("mac") or event.get("client"),
        "name": event.get("name") or event.get("hostname"),
    }


@server.tool(
    name="unifi_network_diagnosis",
    description=(
        "One-shot network health check. Returns system info, network health, "
        "active alarms, recent events, and device summary in a single call. "
        "Use this as the first diagnostic tool — it replaces calling "
        "unifi_get_system_info, unifi_get_network_health, unifi_list_alarms, "
        "unifi_list_events, and unifi_list_devices individually. "
        "Events/alarms are capped at 10; set include_device_list=true for "
        "full device inventory. For deeper investigation, follow up with "
        "the individual tools."
    ),
)
async def network_diagnosis(include_device_list: bool = False) -> Dict[str, Any]:
    """Composite network diagnosis — parallel fetch of key health data."""
    logger.info("unifi_network_diagnosis tool called")

    try:
        event_manager = _get_event_manager()
        alarm_coro = event_manager.get_alarms(archived=False)
        event_coro = event_manager.get_events(within=24)
    except Exception as exc:
        logger.error(f"Failed to initialise event manager: {exc}")

        async def _failed(err):
            raise err

        alarm_coro = _failed(exc)
        event_coro = _failed(exc)

    # Fetch all data sources in parallel; individual failures degrade gracefully
    results = await asyncio.gather(
        system_manager.get_system_info(),
        system_manager.get_network_health(),
        alarm_coro,
        event_coro,
        device_manager.get_devices(),
        return_exceptions=True,
    )

    system_info, network_health, alarms, events, devices = results

    response: Dict[str, Any] = {"success": True}

    # System info (small payload, include fully)
    if isinstance(system_info, Exception):
        logger.error(f"Failed to fetch system info: {system_info}")
        response["system_info"] = {"error": str(system_info)}
    else:
        response["system_info"] = system_info

    # Network health (small payload, include fully)
    if isinstance(network_health, Exception):
        logger.error(f"Failed to fetch network health: {network_health}")
        response["network_health"] = {"error": str(network_health)}
    else:
        response["network_health"] = network_health

    # Alarms — cap at 10, flag truncation
    if isinstance(alarms, Exception):
        logger.error(f"Failed to fetch alarms: {alarms}")
        response["alarms"] = {"error": str(alarms), "count": 0, "items": []}
    else:
        alarm_list = list(alarms) if alarms else []
        response["alarms"] = {
            "count": len(alarm_list),
            "items": alarm_list[:10],
            "truncated": len(alarm_list) > 10,
        }

    # Events — cap at 10, slim fields, flag truncation
    if isinstance(events, Exception):
        logger.error(f"Failed to fetch events: {events}")
        response["events"] = {"error": str(events), "count": 0, "items": []}
    else:
        event_list = list(events) if events else []
        response["events"] = {
            "count": len(event_list),
            "items": [_slim_event(e) for e in event_list[:10]],
            "truncated": len(event_list) > 10,
        }

    # Devices — summary by default, full list if requested
    if isinstance(devices, Exception):
        logger.error(f"Failed to fetch devices: {devices}")
        response["device_summary"] = {"error": str(devices), "total": 0}
    else:
        devices_raw = [d.raw if hasattr(d, "raw") else d for d in devices]
        response["device_summary"] = _summarize_devices(devices_raw, include_device_list)

    return response

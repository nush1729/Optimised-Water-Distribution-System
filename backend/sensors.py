"""Sensor ingest: the latest pressure reading per node, per network.

Readings arrive over HTTP (POST /sensors/readings) or, optionally, MQTT. Anything
that can publish JSON can feed the detector: a SCADA gateway, a LoRaWAN network
server webhook, a Raspberry Pi. This module only stores and serves the latest
snapshot; detection and response live in anomaly/ and main.py.

MQTT (optional, needs `pip install paho-mqtt`):
    HARP_MQTT_URL=mqtt://broker:1883   topic: harp/<network>/pressure
    payload: {"<node_id>": <pressure_m>, ...}
"""
import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_LATEST: dict = {}          # network -> {"ts": float, "readings": {node: pressure_m}}
MAX_READINGS_PER_POST = 20000


def store_readings(network: str, readings: dict, ts: float = None) -> dict:
    """Merge `readings` into the network's latest snapshot. Returns the snapshot."""
    clean = {}
    for node, value in readings.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v != v or v in (float("inf"), float("-inf")):   # NaN / inf
            continue
        clean[str(node)] = v
    with _lock:
        snap = _LATEST.setdefault(network, {"ts": 0.0, "readings": {}})
        snap["readings"].update(clean)
        snap["ts"] = ts if ts is not None else time.time()
        return {"ts": snap["ts"], "readings": dict(snap["readings"])}


def latest(network: str):
    with _lock:
        snap = _LATEST.get(network)
        return None if snap is None else {"ts": snap["ts"], "readings": dict(snap["readings"])}


def clear(network: str = None) -> None:
    with _lock:
        if network is None:
            _LATEST.clear()
        else:
            _LATEST.pop(network, None)


def parse_mqtt_message(topic: str, payload: bytes):
    """harp/<network>/pressure + JSON body -> (network, readings) or None if malformed."""
    parts = topic.split("/")
    if len(parts) != 3 or parts[0] != "harp" or parts[2] != "pressure":
        return None
    try:
        body = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict):
        return None
    return parts[1], body


def start_mqtt_listener_if_configured() -> bool:
    """Start a background MQTT subscriber when HARP_MQTT_URL is set. Returns True if started."""
    url = os.environ.get("HARP_MQTT_URL")
    if not url:
        return False
    try:
        import paho.mqtt.client as mqtt          # optional dependency
        from urllib.parse import urlparse
    except ImportError:
        logger.warning("HARP_MQTT_URL is set but paho-mqtt is not installed; sensor MQTT disabled")
        return False
    u = urlparse(url)

    def on_message(_client, _userdata, msg):
        parsed = parse_mqtt_message(msg.topic, msg.payload)
        if parsed:
            store_readings(*parsed)
        else:
            logger.warning("Ignoring malformed MQTT message on %s", msg.topic)

    client = mqtt.Client()
    if u.username:
        client.username_pw_set(u.username, u.password)
    client.on_message = on_message
    client.connect(u.hostname, u.port or 1883, keepalive=60)
    client.subscribe("harp/+/pressure")
    client.loop_start()
    logger.info("MQTT sensor listener connected to %s", u.hostname)
    return True

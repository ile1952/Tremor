# Bluetooth Data Protocol

## Status

Not defined yet. The ESP32 firmware and MAUI receiver are both planned; agree
on this interface before implementing either Bluetooth endpoint.

## Decisions to make

- Bluetooth Low Energy (BLE) or Bluetooth Classic, based on the ESP32 board and
  the target Mac/iPhone platforms.
- Device name, service UUID, and characteristic UUIDs (for BLE).
- Sample rate, sensor axes, measurement units, and numeric precision.
- Packet layout, sequence numbering, and timestamps.
- Framing and behavior when a packet is lost or a connection is interrupted.
- Device discovery, connection, and reconnect behavior.
- Whether the receiver may write control/configuration commands to the ESP32.

Keep the wire format versioned and document example packets here once agreed.

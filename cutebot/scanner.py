import asyncio
from bleak import BleakScanner

seen = {}  # addr -> (name, rssi, uuids)

def on_detect(device, adv_data):
    name = device.name or getattr(adv_data, "local_name", None) or "(no name)"
    uuids = list(getattr(adv_data, "service_uuids", []) or [])
    rssi = getattr(adv_data, "rssi", None)
    seen[device.address] = (name, rssi, uuids)

async def main():
    print("Scanning for 8s...")
    scanner = BleakScanner(on_detect)
    await scanner.start()
    await asyncio.sleep(8)
    await scanner.stop()

    for addr, (name, rssi, uuids) in seen.items():
        print(f"{name:30} {addr:>24}  RSSI={str(rssi):>4}  {uuids}")

    # Look for micro:bit by name or Nordic UART service UUID
    UART_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
    candidates = [
        (addr, name) for addr, (name, _rssi, uuids) in seen.items()
        if ("micro:bit" in name) or (UART_UUID in uuids)
    ]
    if candidates:
        print("\nLikely micro:bit(s):")
        for addr, name in candidates:
            print(f" - {name}  {addr}")
    else:
        print("\nNo obvious micro:bit found. Put it in pairing mode and scan again.")

asyncio.run(main())

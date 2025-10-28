import asyncio
from bleak import BleakScanner, BleakClient

async def main():
    print("Scanning for micro:bit…")
    dev = None
    async with BleakScanner() as scanner:
        await asyncio.sleep(5)
        for d in scanner.discovered_devices:
            if "micro:bit" in (d.name or ""):
                dev = d
                break
    if not dev:
        print("No micro:bit found.")
        return

    print(f"Connecting to {dev.name} ({dev.address})…")
    async with BleakClient(dev) as client:
        print("Connected. Resolving services…")

        # Bleak compatibility (some versions use .services, some have get_services())
        services = getattr(client, "services", None)
        if services is None or len(list(services.services.keys())) == 0:
            get_services = getattr(client, "get_services", None)
            if callable(get_services):
                services = await get_services()
            else:
                # Fallback: refresh the services cache by reading MTU (forces discovery on some backends)
                try:
                    await client.get_mtu()  # may be a no-op depending on backend
                except Exception:
                    pass
                services = client.services  # try again

        for s in services:
            print(f"SERVICE {s.uuid}")
            for c in s.characteristics:
                props = ",".join(c.properties)
                print(f"  CHAR {c.uuid} [{props}]")

asyncio.run(main())

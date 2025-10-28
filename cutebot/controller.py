import asyncio
from bleak import BleakScanner, BleakClient

UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX      = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify (micro:bit -> host)
UART_RX      = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write (host -> micro:bit)

async def main():
    print("Scanning for BBC micro:bit…")
    dev = None
    async with BleakScanner() as scanner:
        await asyncio.sleep(5)
        for d in scanner.discovered_devices:
            # Don't rely only on name; macOS often hides 128-bit UUIDs in adv
            if d.name and "micro:bit" in d.name:
                dev = d; break
    if not dev:
        print("No micro:bit found. Try pairing mode or disable pairing in MakeCode.")
        return

    print(f"Connecting to {dev.name} ({dev.address})…")
    async with BleakClient(dev, use_cached=False, timeout=15.0) as client:
        print("Connected.")

        def on_notify(_, data: bytearray):
            print("BOT:", data.decode(errors="ignore").strip())

        # listen to TX (notify)
        await client.start_notify(UART_TX, on_notify)

        async def send(cmd: str):
            # write to RX
            await client.write_gatt_char(UART_RX, (cmd + "\n").encode(), response=False)

        print("Ready. Try: V,50,50  |  T,50,0,1000  |  S  |  quit")
        try:
            while True:
                cmd = input("> ").strip()
                if not cmd: continue
                if cmd.lower() == "quit":
                    await send("S"); break
                await send(cmd)
        finally:
            try: await client.stop_notify(UART_TX)
            except: pass

asyncio.run(main())
    
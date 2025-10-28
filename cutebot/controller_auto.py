# controller_auto.py
import asyncio
from bleak import BleakScanner, BleakClient

UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"

async def find_microbit():
    print("Scanning for BBC micro:bit…")
    dev = None
    async with BleakScanner() as scanner:
        await asyncio.sleep(5)
        for d in scanner.discovered_devices:
            if (d.name or "").startswith("BBC micro:bit"):
                dev = d
                break
    return dev

async def resolve_services(client: BleakClient):
    """
    Compatibility shim:
    - Prefer client.services when available.
    - If empty, try client.get_services() when present.
    - If still empty, poke the stack (read MTU) and re-check.
    """
    services = getattr(client, "services", None)

    def _is_empty(coll):
        try:
            # BleakGATTServiceCollection has .services dict in many versions
            return coll is None or (hasattr(coll, "services") and not coll.services)
        except Exception:
            return False

    if _is_empty(services):
        get_services = getattr(client, "get_services", None)
        if callable(get_services):
            try:
                services = await get_services()
            except Exception:
                services = getattr(client, "services", None)

    if _is_empty(services):
        # nudge discovery on some backends
        try:
            await client.get_mtu()  # no-op on some platforms
        except Exception:
            pass
        services = getattr(client, "services", services)

    return services

async def main():
    dev = await find_microbit()
    if not dev:
        print("No micro:bit found.")
        return

    print(f"Connecting to {dev.name} ({dev.address})…")
    async with BleakClient(dev, use_cached=False, timeout=15) as client:
        print("Connected. Resolving services…")
        svcs = await resolve_services(client)
        if not svcs:
            print("Could not resolve services.")
            return

        # Gather UART characteristics regardless of Bleak version
        uart_chars = []
        for s in svcs:
            if s.uuid.lower() == UART_SERVICE:
                for c in s.characteristics:
                    uart_chars.append(c)

        if len(uart_chars) < 2:
            print("UART service not found.")
            # Debug dump of what we *did* see
            for s in svcs:
                print("SVC", s.uuid)
                for c in s.characteristics:
                    print("  CHAR", c.uuid, c.properties)
            return

        # Identify RX (write) and TX (notify/indicate) by properties
        rx_char = next((c for c in uart_chars if "write" in c.properties), None)
        tx_char = next((c for c in uart_chars if ("notify" in c.properties) or ("indicate" in c.properties)), None)

        if not (rx_char and tx_char):
            print("Could not identify RX/TX UART characteristics from properties.")
            for c in uart_chars:
                print(c.uuid, c.properties)
            return

        print(f"UART RX (write to this): {rx_char.uuid}  props={rx_char.properties}")
        print(f"UART TX (listen to this): {tx_char.uuid}  props={tx_char.properties}")

        def on_uart(_, data: bytearray):
            print("BOT:", data.decode(errors="ignore").strip())

        # CoreBluetooth sometimes only offers 'indicate'; Bleak maps start_notify to both
        await client.start_notify(tx_char, on_uart)

        async def send(line: str):
            payload = (line + "\n").encode()
            # If both write and write-without-response exist, prefer without response
            need_response = ("write" in rx_char.properties) and ("write-without-response" not in rx_char.properties)
            await client.write_gatt_char(rx_char, payload, response=need_response)

        print("Ready. Try: V,50,50   |  T,50,0,800   |  S   |  quit")
        try:
            while True:
                cmd = input("> ").strip()
                if not cmd:
                    continue
                if cmd.lower() == "quit":
                    await send("S")
                    break
                await send(cmd)
        finally:
            try:
                await client.stop_notify(tx_char)
            except:
                pass

asyncio.run(main())

# controller_auto.py
import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Callable, Optional

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice

UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"


async def find_microbit(name_prefix: str = "BBC micro:bit") -> Optional[BLEDevice]:
    """
    Scan for a BBC micro:bit advertising the Cutebot UART service.
    """
    print("Scanning for BBC micro:bit…")
    async with BleakScanner() as scanner:
        await asyncio.sleep(5)
        for dev in scanner.discovered_devices:
            if (dev.name or "").startswith(name_prefix):
                return dev
    return None


async def resolve_services(client: BleakClient):
    """
    Compatibility shim that handles Bleak differences across platforms/versions.
    """
    services = getattr(client, "services", None)

    def _is_empty(coll):
        try:
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
        try:
            await client.get_mtu()
        except Exception:
            pass
        services = getattr(client, "services", services)

    return services


def _resolve_uart_characteristics(services):
    uart_chars = []
    for svc in services:
        if svc.uuid.lower() == UART_SERVICE:
            uart_chars.extend(svc.characteristics)

    if len(uart_chars) < 2:
        return None, None

    rx_char = next((c for c in uart_chars if "write" in c.properties), None)
    tx_char = next(
        (
            c
            for c in uart_chars
            if ("notify" in c.properties) or ("indicate" in c.properties)
        ),
        None,
    )
    return rx_char, tx_char


class CutebotUARTSession(AbstractAsyncContextManager):
    """
    Convenience wrapper around Bleak to manage Cutebot UART commands.
    """

    def __init__(
        self,
        device: Optional[BLEDevice] = None,
        *,
        message_handler: Optional[Callable[[str], None]] = None,
        timeout: float = 15.0,
        verbose: bool = True,
        name_prefix: str = "BBC micro:bit",
    ):
        self._device = device
        self._timeout = timeout
        self._message_handler = message_handler
        self._verbose = verbose
        self._name_prefix = name_prefix
        self._client: Optional[BleakClient] = None
        self._rx_char = None
        self._tx_char = None
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        self._send_lock = asyncio.Lock()
        self._notifications_started = False
        self._heading_streaming = False
        self._last_heading: Optional[float] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()
        return False

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg)

    def _handle_uart(self, _sender, data: bytearray) -> None:
        text = data.decode(errors="ignore").strip()
        if not text:
            return
        if text.upper().startswith("HEADING"):
            parts = text.split(",", 1)
            if len(parts) == 2:
                try:
                    self._last_heading = float(parts[1])
                except ValueError:
                    pass
        if self._message_handler:
            self._message_handler(text)
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            try:
                _ = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(text)

    async def connect(self) -> None:
        if self._client and self._client.is_connected:
            return

        device = self._device or await find_microbit(self._name_prefix)
        if not device:
            raise RuntimeError("No micro:bit found.")

        self._log(f"Connecting to {device.name} ({device.address})…")
        client = BleakClient(device, use_cached=False, timeout=self._timeout)
        await client.connect()
        self._client = client

        self._log("Connected. Resolving services…")
        services = await resolve_services(client)
        if not services:
            raise RuntimeError("Could not resolve services.")

        rx_char, tx_char = _resolve_uart_characteristics(services)
        if not (rx_char and tx_char):
            raise RuntimeError("UART service not found on the connected device.")

        self._rx_char = rx_char
        self._tx_char = tx_char
        await client.start_notify(tx_char, self._handle_uart)
        self._notifications_started = True
        self._log(
            f"UART ready. RX={rx_char.uuid} props={rx_char.properties} | "
            f"TX={tx_char.uuid} props={tx_char.properties}"
        )

    async def disconnect(self) -> None:
        if not self._client:
            return

        try:
            if self._notifications_started and self._tx_char:
                await self._client.stop_notify(self._tx_char)
        except Exception:
            pass
        finally:
            self._notifications_started = False

        try:
            if self._client.is_connected:
                await self._client.disconnect()
        finally:
            self._client = None
            self._rx_char = None
            self._tx_char = None

    async def send_line(self, line: str) -> None:
        if not (self._client and self._client.is_connected and self._rx_char):
            raise RuntimeError("Cutebot not connected.")

        payload = (line.strip() + "\n").encode()
        need_response = ("write" in self._rx_char.properties) and (
            "write-without-response" not in self._rx_char.properties
        )
        async with self._send_lock:
            await self._client.write_gatt_char(
                self._rx_char, payload, response=need_response
            )

    @staticmethod
    def _clamp_speed(value: int) -> int:
        return max(0, min(100, int(value)))

    async def drive_timed(
        self,
        left: int,
        right: int,
        duration_ms: int,
        *,
        wait: bool = True,
        settle_sec: float = 0.15,
    ) -> None:
        """
        Issue a timed drive command (Cutebot micro:bit expects T,left,right,duration_ms).
        """
        cl = self._clamp_speed(left)
        cr = self._clamp_speed(right)
        await self.send_line(f"T,{cl},{cr},{int(duration_ms)}")
        if wait:
            await asyncio.sleep(max(0.0, duration_ms / 1000.0) + settle_sec)

    async def set_velocity(self, left: int, right: int) -> None:
        cl = self._clamp_speed(left)
        cr = self._clamp_speed(right)
        await self.send_line(f"V,{cl},{cr}")

    async def stop(self) -> None:
        await self.send_line("S")

    async def get_notification(self, timeout: Optional[float] = None) -> Optional[str]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def enable_heading_stream(self, enable: bool = True) -> None:
        command = "H,ON" if enable else "H,OFF"
        await self.send_line(command)
        self._heading_streaming = enable

    async def request_heading(self, timeout: float = 2.0) -> Optional[float]:
        await self.send_line("H")
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            msg = await self.get_notification(timeout=remaining)
            if not msg:
                continue
            if msg.upper().startswith("HEADING"):
                parts = msg.split(",", 1)
                if len(parts) == 2:
                    try:
                        self._last_heading = float(parts[1])
                        return self._last_heading
                    except ValueError:
                        continue
        return None

    @property
    def last_heading(self) -> Optional[float]:
        return self._last_heading



async def interactive_cli() -> None:
    session = CutebotUARTSession(message_handler=lambda msg: print("BOT:", msg))
    try:
        async with session:
            print("Ready. Try: V,50,50   |  T,50,0,800   |  S   |  quit")
            while True:
                try:
                    cmd = (await asyncio.to_thread(input, "> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not cmd:
                    continue
                if cmd.lower() == "quit":
                    await session.stop()
                    break
                await session.send_line(cmd)
    except RuntimeError as exc:
        print(exc)
    finally:
        await session.disconnect()


if __name__ == "__main__":
    asyncio.run(interactive_cli())

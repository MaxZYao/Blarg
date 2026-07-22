from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import numpy as np

from station import RealSerial, Link, SensorClient
from station.protocol import ProtocolError


def _ping(link, attempts: int = 8, timeout: float = 0.8) -> bool:
    """Ping with short retries so the board's ~1.7 s boot setup doesn't fail us."""
    for _ in range(attempts):
        try:
            r = link.command("PING", timeout=timeout)
            if r.ok and r.payload == "PONG":
                return True
        except TimeoutError:
            pass
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Log the Teensy sensor pipeline to CSV.")
    ap.add_argument("--port", help="Teensy COM port, e.g. COM7 (omit to list)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--hz", type=float, default=5.0, help="frames per second")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds (default: until Ctrl-C)")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: logs/teensy_<timestamp>.csv)")
    args = ap.parse_args()

    from serial_ports import resolve
    args.port = resolve(args.port)
    if not args.port:
        return 1

    try:
        link = Link(RealSerial(args.port, args.baud), "sensor")
    except Exception as e:  # noqa: BLE001
        print(f"Could not open {args.port}: {e}")
        return 1
    sensors = SensorClient(link)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (Path(args.out) if args.out
           else Path(__file__).parent / "logs" / f"teensy_{ts}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("PING (waiting for the Teensy)...", end="", flush=True)
    if not _ping(link):
        print(" NO RESPONSE.")
        print("\nThe firmware compiles/flashes fine, so this is a connection issue:")
        print("  - correct COM port? (this one:", args.port + ")")
        print("  - is the Arduino Serial Monitor still open? (it holds the port)")
        print("  - is the Teensy powered and running sensor_teensy.ino?")
        link.close()
        return 1
    print(" PONG")
    try:
        print("STATUS:", sensors.status())
    except (TimeoutError, ProtocolError):
        pass

    n_units = sensors.n_units() or 8
    period = 1.0 / args.hz
    deadline = (time.monotonic() + args.duration) if args.duration else None
    print(f"Logging {n_units}-DUT frames to {out}   (Ctrl-C to stop)")
    print("-" * 70)

    frame_idx = 0
    rows = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iso_time", "frame", "t_us", "enc_deg", "temp_uut",
                    "temp_ref", "dut", "ax_g", "ay_g", "az_g", "mag_g",
                    "pitch_deg", "roll_deg"])
        try:
            while deadline is None or time.monotonic() < deadline:
                start = time.monotonic()
                try:
                    f = sensors.read_frame()
                except (ProtocolError, TimeoutError) as e:
                    print(f"[{frame_idx}] read failed: {e}")
                    time.sleep(period)
                    continue

                stamp = datetime.datetime.now().isoformat()
                t_uut = float(f.temp[0]) if f.temp.size else float("nan")
                t_ref = float(f.temp[1]) if f.temp.size > 1 else float("nan")
                for u in range(f.units.shape[0]):
                    ax, ay, az = (float(x) for x in f.units[u])
                    mag = float(np.linalg.norm(f.units[u]))
                    p = (float(f.units_pr[u][0]) if f.units_pr is not None else "")
                    r = (float(f.units_pr[u][1]) if f.units_pr is not None else "")
                    w.writerow([stamp, frame_idx, f.t_us, f"{f.enc_deg:.4f}",
                                f"{t_uut:.3f}", f"{t_ref:.3f}", u,
                                f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}", f"{mag:.6f}",
                                (f"{p:.2f}" if p != "" else ""),
                                (f"{r:.2f}" if r != "" else "")])
                    rows += 1
                fh.flush()

                live = (int(f.units_valid.sum()) if f.units_valid is not None
                        else sum(1 for u in range(f.units.shape[0])
                                 if np.linalg.norm(f.units[u]) > 1e-6))
                if frame_idx < 10 or frame_idx % 10 == 0:
                    print(f"[{frame_idx:5d}] enc={f.enc_deg:+7.2f} deg   "
                          f"{live}/{f.units.shape[0]} DUTs live   "
                          f"DUT0 |a|={np.linalg.norm(f.units[0]):.4f} g")
                frame_idx += 1
                time.sleep(max(0.0, period - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass

    print("-" * 70)
    print(f"Stopped. {frame_idx} frames ({rows} rows) saved to {out}")
    link.close()
    return 0


if __name__ == "__main__":
sys.exit(main())

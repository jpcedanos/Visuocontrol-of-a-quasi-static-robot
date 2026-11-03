"""Print the stream from hand_detector.py without running the Processing sketch.

Usage:
    python debug_client.py [--host 127.0.0.1] [--port 65432]
"""

import argparse
import socket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port)) as conn, conn.makefile("r") as stream:
        print(f"Connected to {args.host}:{args.port}")
        for line in stream:
            x, y, z, grab = line.strip().split(",")
            print(f"x={float(x):7.2f}  y={float(y):7.2f}  z={float(z):7.2f}  grab={grab}")
    print("Server closed the connection")


if __name__ == "__main__":
    main()

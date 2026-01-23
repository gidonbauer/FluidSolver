#!/usr/bin/python3

import sys
import os


def file_needs_patching(filename: str) -> bool:
    try:
        with open(filename, 'rb') as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError as err:
                print(f"{err}")
                sys.exit(1)
            last_line = f.readline().decode()
            return not "</Xdmf>" in last_line
    except IOError as err:
        print(f"{err}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <XDMF file>", file=sys.stderr)
        sys.exit(1)
    
    filename = sys.argv[1]
    if not file_needs_patching(filename):
        sys.exit(0)

    try:
        with open(filename, "a") as f:
            print("    </Grid>", file=f)
            print("  </Domain>", file=f)
            print("</Xdmf>", file=f)
    except IOError as err:
        print(f"{err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

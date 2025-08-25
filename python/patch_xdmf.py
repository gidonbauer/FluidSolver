#!/usr/bin/python3

import sys


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <XDMF file>", file=sys.stderr)
        sys.exit(1)
    
    filename = sys.argv[1]
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
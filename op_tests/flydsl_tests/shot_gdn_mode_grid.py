# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-page screenshot of a rendered mode grid, for embedding in the doc.

The page is the deliverable; this only produces the static copy a markdown
document can inline.  Width is fixed so the grid does not reflow between runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1460)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    src = Path(args.html).resolve()
    if not src.exists():
        raise SystemExit(f"no such page: {src}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": 1200},
            device_scale_factor=2,
        )
        page.goto(src.as_uri())
        page.wait_for_load_state("networkidle")
        page.screenshot(path=args.out, full_page=True)
        browser.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

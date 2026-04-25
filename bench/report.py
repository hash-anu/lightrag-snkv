"""Render benchmark results as a text table and optionally as Markdown."""
from __future__ import annotations

from typing import Sequence

from bench.measure import Measurement


def print_report(
    measurements: Sequence[Measurement],
    percentiles: list[int] | None = None,
    markdown: bool = False,
) -> None:
    if percentiles is None:
        percentiles = [50, 95, 99]

    # Group by mode
    modes = list(dict.fromkeys(m.mode for m in measurements))
    stacks = list(dict.fromkeys(m.stack for m in measurements))

    for mode in modes:
        mode_ms = {
            m.stack: m
            for m in measurements
            if m.mode == mode
        }

        header_cols = ["Stack", "N", "Mean(ms)"] + [f"p{p}(ms)" for p in percentiles]
        rows = []
        for stack in stacks:
            if stack not in mode_ms:
                continue
            m = mode_ms[stack]
            row = [
                stack,
                str(len(m.samples_ms)),
                f"{m.mean():.1f}",
            ] + [f"{m.percentile(p):.1f}" for p in percentiles]
            rows.append(row)

        col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header_cols)]

        sep = "  "
        if markdown:
            print(f"\n### Mode: `{mode}`\n")
            header = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header_cols)) + " |"
            divider = "| " + " | ".join("-" * col_widths[i] for i in range(len(header_cols))) + " |"
            print(header)
            print(divider)
            for row in rows:
                print("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |")
        else:
            print(f"\nMode: {mode}")
            print(sep.join(h.ljust(col_widths[i]) for i, h in enumerate(header_cols)))
            print(sep.join("-" * col_widths[i] for i in range(len(header_cols))))
            for row in rows:
                print(sep.join(row[i].ljust(col_widths[i]) for i in range(len(row))))

        # Speedup relative to nano (baseline)
        if "nano" in mode_ms and len(stacks) > 1:
            nano_p50 = mode_ms["nano"].percentile(50)
            if nano_p50 > 0:
                print(f"\n  Speedup vs nano (p50):")
                for stack in stacks:
                    if stack == "nano" or stack not in mode_ms:
                        continue
                    snkv_p50 = mode_ms[stack].percentile(50)
                    speedup = nano_p50 / snkv_p50 if snkv_p50 > 0 else float("inf")
                    print(f"    {stack}: {speedup:.2f}x")

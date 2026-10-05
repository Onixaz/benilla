#!/usr/bin/env python3
"""Cast-to-cast cycle times from a WOW_MOVE_TRACE of a mashed cast, for the spell queue.

A probe mashes one spell every frame (WOW_PROBE_LUA calling CastSpellByName from an OnUpdate), then
another from `split` seconds after the first send; the trace keeps the `out` tag. Each phase reports
its send-to-send cycle and the dead time past the nominal cycle (the cast time, or the 1.5 s GCD for
an instant). Sends alone are timed, so a reply the early send overtook cannot be misattributed.

  WOW_MOVE_TRACE=cast.trace WOW_MOVE_TRACE_TAGS=in,out WOW_PROBE_LUA='...' cargo play
  scripts/castcycles.py cast.trace 29.5 [nominal_ms=1500]

Rejections are approximate: CAST_RESULT lines beyond SPELL_GO lines (a refusal has no GO).
"""
import re
import statistics as st
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, split = sys.argv[1], float(sys.argv[2])
    nominal = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    send = re.compile(r"^t=\s*([\d.]+)\s+out\s+\S+\s+CMSG_CAST_SPELL")
    sends = [float(m.group(1)) for line in lines if (m := send.match(line))]
    if not sends:
        sys.exit(f"{path}: no CMSG_CAST_SPELL sends (was the `out` tag kept?)")
    rejected = sum("SMSG_CAST_RESULT" in l for l in lines) - sum("SMSG_SPELL_GO" in l for l in lines)
    t0 = sends[0]
    for name, lo, hi in (("phase 1", 0.0, split), ("phase 2", split, float("inf"))):
        s = [t for t in sends if lo <= t - t0 < hi]
        cycles = sorted((b - a) * 1000 for a, b in zip(s, s[1:]))
        if not cycles:
            continue
        p95 = cycles[min(len(cycles) - 1, int(round(0.95 * (len(cycles) - 1))))]
        med = st.median(cycles)
        print(
            f"{name}  n={len(cycles):2}  cycle median={med:6.0f}  min={cycles[0]:6.0f}  "
            f"p95={p95:6.0f}  max={cycles[-1]:6.0f}  dead median={med - nominal:5.0f} ms"
        )
    print(f"rejections (CAST_RESULT minus SPELL_GO, approximate): {rejected}")


if __name__ == "__main__":
    main()

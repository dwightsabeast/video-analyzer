"""Binomial "luck" analysis for run verification (Dream-report style).

Cheated RNG shows up as event streaks far beyond plausible luck. For each
event class the moderator records how many trials are visible in the video,
how many succeeded, and the per-trial probability from the game's data
tables; this module turns that into exact binomial upper-tail odds, applies
the selection correction popularised by the Dream investigation (the runner
and run under review were *picked* because they looked lucky, out of many
candidates), and renders a moderator-readable report.

Stdlib only. Tails are summed in log space so "one in trillions" keeps
precision where naive float products underflow. Luck is evidence, never
proof - the report says so."""

from __future__ import annotations

import math

_LOG10 = math.log(10.0)


def _log_pmf(n: int, k: int, lp: float, lq: float) -> float:
    """ln of the Binomial(n, p) pmf at k, via lgamma."""
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * lp + (n - k) * lq)


def _norm_sf_log10(z: float) -> float:
    """log10 of the standard-normal upper tail; asymptotic form once erfc
    underflows (z > ~37)."""
    if z < 30.0:
        v = 0.5 * math.erfc(z / math.sqrt(2.0))
        if v > 0.0:
            return math.log10(v)
    return (-(z * z) / 2.0 - math.log(z * math.sqrt(2.0 * math.pi))) / _LOG10


def binom_tail_log10(n: int, k: int, p: float) -> float:
    """log10 of P(X >= k) for X ~ Binomial(n, p).

    Exact term-ratio summation of the upper tail; falls back to a normal
    approximation only for astronomically large n (> 5e7 trials). Returns
    0.0 (= log10 of 1) when k <= 0 and -inf when k > n."""
    n, k = int(n), int(k)
    if k <= 0:
        return 0.0
    if k > n:
        return -math.inf
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return 0.0
    if n > 50_000_000:                      # keep the loop bounded
        mu, sd = n * p, math.sqrt(n * p * (1.0 - p))
        return _norm_sf_log10((k - 0.5 - mu) / sd)
    lp, lq = math.log(p), math.log1p(-p)
    log_t0 = _log_pmf(n, k, lp, lq)
    ratio = p / (1.0 - p)
    acc, term = 1.0, 1.0
    for i in range(k, n):
        term *= (n - i) / (i + 1.0) * ratio
        acc += term
        if term < acc * 1e-18 and i > n * p:
            break
    return min(0.0, (log_t0 + math.log(acc)) / _LOG10)


def one_in(log10p: float) -> str:
    """Human phrasing of a log10 probability: '1 in 7.5 trillion'."""
    if log10p == -math.inf:
        return "impossible (successes exceed trials)"
    if log10p >= -0.3011:
        return "about 1 in %.1f" % (10.0 ** -log10p)
    x = -log10p
    names = [(3, "thousand"), (6, "million"), (9, "billion"), (12, "trillion"),
             (15, "quadrillion"), (18, "quintillion")]
    for exp, name in reversed(names):
        if x >= exp and x < exp + 3:
            return "1 in %.1f %s" % (10.0 ** (x - exp), name)
    if x < 3:
        return "1 in %d" % round(10.0 ** x)
    return "1 in 10^%.1f" % x


def evaluate(events: "list[dict]") -> dict:
    """events: [{label, n, k, p, select?}] where n = visible trials,
    k = successes, p = per-trial probability from game data, select = how
    many comparable runs/runners the reviewer could have picked from
    (selection / cherry-pick correction; default 1 = none).

    Returns per-event and combined log10 tails, raw and corrected, with a
    cautious verdict band."""
    rows = []
    total_raw = total_adj = 0.0
    for ev in events:
        n, k, p = int(ev["n"]), int(ev["k"]), float(ev["p"])
        sel = max(1, int(ev.get("select") or 1))
        raw = binom_tail_log10(n, k, p)
        adj = min(0.0, raw + math.log10(sel))
        rows.append({"label": str(ev.get("label") or "event"),
                     "n": n, "k": k, "p": p, "select": sel,
                     "log10_raw": raw, "log10_adj": adj,
                     "one_in_raw": one_in(raw), "one_in_adj": one_in(adj)})
        total_raw += raw
        total_adj += adj
    if total_adj < -9.0:
        verdict, note = "implausible", "far beyond any reasonable luck"
    elif total_adj < -5.0:
        verdict, note = "suspicious", "needs corroborating evidence"
    else:
        verdict, note = "unremarkable", "within ordinary luck"
    return {"events": rows,
            "log10_combined_raw": total_raw,
            "log10_combined_adj": total_adj,
            "one_in_combined": one_in(total_adj),
            "verdict": verdict, "verdict_note": note}


def render_report(res: dict) -> str:
    lines = ["RNG LUCK ANALYSIS (binomial upper tails)", ""]
    lines.append("%-22s %8s %8s %10s %7s  %s"
                 % ("event", "trials", "hits", "p(hit)", "pick-of", "odds (corrected)"))
    for r in res["events"]:
        lines.append("%-22s %8d %8d %10.5g %7d  %s   [raw %s]"
                     % (r["label"][:22], r["n"], r["k"], r["p"], r["select"],
                        r["one_in_adj"], r["one_in_raw"]))
    lines += ["",
              "Combined (independent events): %s   (log10 p = %.2f raw, %.2f corrected)"
              % (res["one_in_combined"], res["log10_combined_raw"],
                 res["log10_combined_adj"]),
              "Verdict: %s - %s" % (res["verdict"].upper(), res["verdict_note"]),
              "",
              "Caveats: probabilities must come from verified game data for the",
              "exact version; events are assumed independent; runners who stop",
              "on success bias tails optimistic - raise 'pick-of' to compensate",
              "(streams examined x runners x event windows). Improbable luck is",
              "evidence for a deeper look (RNG mods, splices), never proof."]
    return "\n".join(lines)

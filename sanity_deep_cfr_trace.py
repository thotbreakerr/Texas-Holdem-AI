"""
sanity_deep_cfr_trace.py — regressions for the Deep CFR decision tracer
(Key Change #1).  Diagnostic only: no training/model/checkpoint behavior is
exercised.

Proves:

  1. bot.act(view) and bot.act_with_trace(view) return the SAME action when
     started from the same RNG state — search disabled AND enabled.  (The two
     methods share the single _decide() code path, so this guards against any
     drift between traced and real decisions.)
  2. Trace metadata exposes the selected abstract label and the concrete final
     amount, and they agree with the returned action.
  3. Tracing works with search disabled (probs_after_search is None,
     search_enabled False) and enabled (probs_after_search populated,
     search_enabled True).
  4. The trace path mutates opponent-tracker / history state by exactly the
     same amount act() does (compared on two fresh bots fed a history-bearing
     view).
"""
import io
import random
import sys
from contextlib import redirect_stdout

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

from core.bot_api import PlayerView, Action
from bots.deep_cfr_bot import (
    DeepCFRBot, DeepCFRConfig, ABSTRACT_ACTIONS, _is_effective_all_in,
)


def _make_bot():
    """Small, weightless bot — fast and deterministic given the RNG state.

    No checkpoint is loaded, so the all-in masks / caps that key off a loaded
    iteration stay inert; we only exercise the trace plumbing.
    """
    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=True)
    return bot


def _preflop_view():
    return PlayerView(
        me="hero", street="preflop", position="BTN",
        hole_cards=[("A", "s"), ("K", "s")], board=[],
        pot=15, to_call=10, min_raise=20, max_raise=500,
        legal_actions=[{"type": "fold"}, {"type": "call"},
                       {"type": "raise", "min": 20, "max": 500}],
        stacks={"hero": 500, "opp1": 500, "opp2": 500},
        opponents=["opp1", "opp2"],
        history=[],
    )


def _history_view():
    """HU preflop limp+raise so observe_action() actually mutates the tracker."""
    return PlayerView(
        me="hero", street="preflop", position="BB",
        hole_cards=[("A", "h"), ("K", "h")], board=[],
        pot=80, to_call=40, min_raise=100, max_raise=440,
        legal_actions=[{"type": "fold"}, {"type": "call"},
                       {"type": "raise", "min": 100, "max": 500}],
        stacks={"hero": 480, "opp": 440},
        opponents=["opp"],
        history=[
            {"type": "blind", "pid": "opp", "amount": 10,
             "street": "preflop", "pot_before": 0},
            {"type": "blind", "pid": "hero", "amount": 20,
             "street": "preflop", "pot_before": 10},
            {"type": "raise", "pid": "opp", "amount": 60,
             "street": "preflop", "to_call_before": 10, "pot_before": 30},
        ],
    )


def _same_action(a, b):
    return a.type == b.type and a.amount == b.amount


def _determinism_and_meta(bot, view):
    """Run act() then act_with_trace() from an identical RNG state."""
    rng_state = bot._rng.getstate()
    mod_state = random.getstate()
    a1 = bot.act(view)
    # Restore BOTH RNGs: act()'s sampling draws from bot._rng, while the search
    # path's chance nodes draw from the module-level random.
    bot._rng.setstate(rng_state)
    random.setstate(mod_state)
    a2, trace = bot.act_with_trace(view)
    return a1, a2, trace


def _tracker_sig(bot):
    t = bot._opp_stats
    if t is None:
        current = history_lens = None
    else:
        current = {
            i: (rec.actions_seen, rec.vpip, rec.pfr,
                rec.bets_and_raises, rec.calls)
            for i, rec in t._current.items()
        }
        history_lens = {i: len(dq) for i, dq in t._history.items()}
    return {
        "last_history_len": bot._last_history_len,
        "last_history_snapshot": list(bot._last_history_snapshot),
        "last_hand_id": bot._last_hand_id,
        "current": current,
        "history_lens": history_lens,
    }


def run():
    PASS = True
    view = _preflop_view()

    # ── CHECK 1: search DISABLED — same action, metadata present ─────────────
    bot = _make_bot()
    bot.search_depth = 0  # no weights anyway => search branch skipped
    a1, a2, trace = _determinism_and_meta(bot, view)
    if _same_action(a1, a2):
        print(f"[CHECK 1] PASS — search off: act()={a1.type}/{a1.amount} == "
              f"act_with_trace()={a2.type}/{a2.amount}")
    else:
        PASS = False
        print(f"[CHECK 1] FAIL — search off: act()={a1.type}/{a1.amount} != "
              f"act_with_trace()={a2.type}/{a2.amount}")

    label_ok = (
        trace.selected_abstract_label in ABSTRACT_ACTIONS
        and trace.final_action_type == a2.type
        and trace.final_action_amount == a2.amount
    )
    if label_ok:
        print(f"[CHECK 2] PASS — trace exposes label "
              f"'{trace.selected_abstract_label}' and concrete "
              f"{trace.final_action_type}/{trace.final_action_amount} "
              f"matching the action")
    else:
        PASS = False
        print(f"[CHECK 2] FAIL — trace metadata mismatch: label="
              f"{trace.selected_abstract_label!r} "
              f"final={trace.final_action_type}/{trace.final_action_amount} "
              f"action={a2.type}/{a2.amount}")

    if trace.probs_after_search is None and trace.search_enabled is False:
        print("[CHECK 3] PASS — search off: probs_after_search is None, "
              "search_enabled False")
    else:
        PASS = False
        print(f"[CHECK 3] FAIL — search off but probs_after_search="
              f"{trace.probs_after_search} search_enabled={trace.search_enabled}")

    # ── CHECK 4: search ENABLED — same action, post-search probs present ──────
    bot_s = _make_bot()
    bot_s._weights_loaded = True   # enable the real-time search branch
    bot_s.search_depth = 2
    a1s, a2s, trace_s = _determinism_and_meta(bot_s, view)
    if _same_action(a1s, a2s):
        print(f"[CHECK 4] PASS — search on: act()={a1s.type}/{a1s.amount} == "
              f"act_with_trace()={a2s.type}/{a2s.amount}")
    else:
        PASS = False
        print(f"[CHECK 4] FAIL — search on: act()={a1s.type}/{a1s.amount} != "
              f"act_with_trace()={a2s.type}/{a2s.amount}")

    if trace_s.probs_after_search is not None and trace_s.search_enabled is True:
        print("[CHECK 5] PASS — search on: probs_after_search populated, "
              "search_enabled True")
    else:
        PASS = False
        print(f"[CHECK 5] FAIL — search on but probs_after_search="
              f"{trace_s.probs_after_search} search_enabled={trace_s.search_enabled}")

    # ── CHECK 6: trace mutates tracker/history exactly like act() ────────────
    hist_view = _history_view()
    b_act = _make_bot()
    b_act.act(hist_view)
    sig_act = _tracker_sig(b_act)

    b_trace = _make_bot()
    b_trace.act_with_trace(hist_view)
    sig_trace = _tracker_sig(b_trace)

    if sig_act == sig_trace:
        print(f"[CHECK 6] PASS — trace path mutates tracker/history identically "
              f"to act() (current={sig_act['current']}, "
              f"hist_len={sig_act['last_history_len']})")
    else:
        PASS = False
        print(f"[CHECK 6] FAIL — tracker/history mutation differs:\n"
              f"  act():   {sig_act}\n  trace(): {sig_trace}")

    # ── CHECK 7: the requirement-5 discriminator, incl. the branch the probe
    #            structurally cannot reach (bucket converted to a max raise) ──
    legal_bet = [{"type": "check"}, {"type": "bet", "min": 10, "max": 900}]
    cases = [
        # action, expected effective-all-in
        (Action("all_in", 50), True),                  # literal all_in
        (Action("raise", 900), True),                  # bucket -> max raise
        (Action("bet", 900), True),                    # bucket -> max bet
        (Action("raise", 60), False),                  # normal sub-max raise
        (Action("call"), False),
        (Action("fold"), False),
    ]
    disc_ok = all(
        _is_effective_all_in(act, legal_bet) is expected for act, expected in cases
    )
    if disc_ok:
        print("[CHECK 7] PASS — _is_effective_all_in flags max-stack bet/raise "
              "(bucket->max) and literal all_in, not sub-max/passive actions")
    else:
        PASS = False
        bad = [(a.type, a.amount, exp, _is_effective_all_in(a, legal_bet))
               for a, exp in cases
               if _is_effective_all_in(a, legal_bet) is not exp]
        print(f"[CHECK 7] FAIL — discriminator mismatches (type,amt,exp,got): {bad}")

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if PASS else 'SOME CHECKS FAILED [FAIL]'}")
    return PASS


if __name__ == "__main__":
    sys.exit(0 if run() else 1)

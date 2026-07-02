"""
sanity_review_findings.py -- regression probes for current review findings.

These checks intentionally encode the desired contracts before the engine/CFR
logic fixes are applied. On the current reviewed code, several sections are
expected to fail; after the fixes, this script should pass cleanly.

Sections:
  1. CFR internal legal actions mirror engine matched-bet semantics.
  2. Tabular CFR inference state preserves PlayerView.to_call.
  3. Deep CFR subtracts action costs in the same unit as value targets.
  4. Missing inference artifacts fail loudly and clearly.
  5. Engine exposes short all-in raises below the minimum raise.
  6. CFR states keep callers after short all-ins.
  7. CFR states use last raise size for reraise minimums.
  8. Deep CFR search root preserves engine bet/check roots.
  9. Deep CFR all-in warmup starts staged policy exposure.
"""
from __future__ import annotations

import io
import inspect
import math
import os
import pickle
import sys
import tempfile
import uuid
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bots import create_bot
import bots.cfr_bot as cfr_mod
from bots.cfr_bot import CFRBot, _CFRNode, _GameState, _build_game_state_from_view
from bots.deep_cfr_bot import (
    ABSTRACT_ACTIONS as DEEP_ABSTRACT_ACTIONS,
    NUM_ACTIONS as DEEP_NUM_ACTIONS,
    DeepCFRBot,
    DeepCFRConfig,
    ReservoirBuffer,
    _DeepCFRGameState,
    build_network_input,
)
from core.bot_api import Action, PlayerView
from core.aivat import _side_pot_awards
from core.engine import InProcessBot, Seat, Table
from core.table_order import advance_dealer_seat_index, normalize_dealer_seat_index
import core.tournament as tournament_mod
from bots.icm_bot import icm_equity, icm_ev_of_call
import run_eval
import training.train_deep_cfr as train_deep_cfr


FAILURES: list[str] = []


def check(name: str, condition: bool, details: str) -> None:
    if condition:
        print(f"  PASS - {name}")
    else:
        print(f"  FAIL - {name}: {details}")
        FAILURES.append(f"{name}: {details}")


def action_types(actions: list[dict]) -> set[str]:
    return {a.get("type", "") for a in actions}


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def test_matched_bet_legal_actions() -> None:
    section("Section 1: matched-bet legal actions")

    tabular = _GameState(
        pot=60,
        stacks=[990, 990, 990],
        committed_per_seat=[10, 10, 10],
        alive=[True, True, True],
        street="preflop",
        board=[],
        hole_cards={},
        seat_order=[2],
        big_blind=10,
    )
    tabular_legal = tabular.legal_actions()
    tabular_types = action_types(tabular_legal)
    tabular_raise = next(
        (a for a in tabular_legal if a.get("type") == "raise"), None
    )
    check(
        "tabular CFR uses raise, not bet, after matching current bet",
        "check" in tabular_types
        and "raise" in tabular_types
        and "bet" not in tabular_types
        and tabular_raise is not None
        and tabular_raise.get("min") == 20,
        f"legal={tabular_legal}",
    )

    deep = _DeepCFRGameState(
        pot=60,
        stacks=[990, 990, 990],
        committed_per_seat=[10, 10, 10],
        alive=[True, True, True],
        street="preflop",
        board=[],
        hole_cards={},
        seat_order=[2],
        action_idx=0,
        history_events=[],
        deck_remaining=[],
        big_blind=10,
    )
    deep_legal = deep.legal_actions()
    deep_types = action_types(deep_legal)
    deep_raise = next((a for a in deep_legal if a.get("type") == "raise"), None)
    check(
        "Deep CFR uses raise, not bet, after matching current bet",
        "check" in deep_types
        and "raise" in deep_types
        and "bet" not in deep_types
        and deep_raise is not None
        and deep_raise.get("min") == 20,
        f"legal={deep_legal}",
    )


def test_tabular_inference_to_call() -> None:
    section("Section 2: tabular inference state preserves to_call")

    view = PlayerView(
        me="P1",
        street="preflop",
        position="SB",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[],
        pot=15,
        to_call=10,
        min_raise=20,
        max_raise=995,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 20, "max": 995},
        ],
        stacks={"P1": 995, "P2": 990},
        opponents=["P2"],
        history=[],
    )

    game_state, hero_seat = _build_game_state_from_view(view)
    internal_to_call = game_state.to_call_for(hero_seat)
    internal_legal = game_state.legal_actions()
    internal_types = action_types(internal_legal)

    check(
        "internal to_call matches PlayerView.to_call",
        internal_to_call == view.to_call,
        (
            f"expected {view.to_call}, got {internal_to_call}; "
            f"committed={game_state.committed_per_seat}"
        ),
    )
    check(
        "internal legal actions face the bet",
        {"fold", "call"}.issubset(internal_types)
        and "check" not in internal_types
        and "bet" not in internal_types,
        f"legal={internal_legal}",
    )


def test_deep_cfr_cost_units() -> None:
    section("Section 3: Deep CFR cost/value unit consistency")

    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(
            config=DeepCFRConfig.small(),
            inference_mode=False,
            aivat_sims=1,
        )

    # Force deterministic leaf values and strategy so the collected regret is
    # exactly the action cost unit. With BB-normalized value targets, a 10-chip
    # call at big_blind=10 should contribute -1.0, not -10.0.
    bot._aivat_leaf_value = lambda _state, _hero_seat: 0.0

    def fold_only_strategy(_logits, _legal_mask):
        strategy = [0.0] * DEEP_NUM_ACTIONS
        strategy[DEEP_ABSTRACT_ACTIONS.index("fold")] = 1.0
        return strategy

    bot._regret_match = fold_only_strategy

    state = _DeepCFRGameState(
        pot=10,
        stacks=[10, 90],
        committed_per_seat=[0, 10],
        alive=[True, True],
        street="preflop",
        board=[],
        hole_cards={
            0: (("A", "s"), ("K", "s")),
            1: (("2", "c"), ("7", "d")),
        },
        seat_order=[0],
        action_idx=0,
        history_events=[],
        deck_remaining=[],
        big_blind=10,
        ring_order=[0, 1],
    )
    regret_buf = ReservoirBuffer()
    value_buf = ReservoirBuffer()
    sizing_buf = ReservoirBuffer()
    bot._cfr_recurse(
        state,
        hero_seat=0,
        depth=1,
        iteration=1,
        regret_buf=regret_buf,
        value_buf=value_buf,
        sizing_buf=sizing_buf,
    )

    call_idx = DEEP_ABSTRACT_ACTIONS.index("check_call")
    actual_regret = float(regret_buf.buffer[0][1][call_idx].item())
    check(
        "10-chip call is charged as 1 big blind in BB-normalized traversal",
        math.isclose(actual_regret, -1.0, rel_tol=0.0, abs_tol=1e-6),
        f"expected call regret -1.0, got {actual_regret}",
    )


def _missing_path(suffix: str) -> str:
    name = f"texas_holdem_ai_missing_{uuid.uuid4().hex}{suffix}"
    return os.path.join(tempfile.gettempdir(), name)


def _clear_missing_error(exc: BaseException, path: str) -> bool:
    message = str(exc).lower()
    return (
        isinstance(exc, (RuntimeError, ValueError))
        and path in str(exc)
        and any(token in message for token in ("missing", "not found", "train"))
    )


def expect_missing_artifact_error(spec: str, path: str) -> tuple[bool, str]:
    try:
        create_bot(spec)
    except Exception as exc:  # noqa: BLE001 - sanity script reports exact failure.
        if _clear_missing_error(exc, path):
            return True, f"{type(exc).__name__}: {exc}"
        return (
            False,
            (
                f"expected RuntimeError/ValueError with clear missing-artifact "
                f"message, got {type(exc).__name__}: {exc}"
            ),
        )
    return False, "factory returned a bot instead of failing"


def test_missing_artifact_loading() -> None:
    section("Section 4: missing inference artifacts fail clearly")

    cfr_path = _missing_path(".pkl")
    cfr_ok, cfr_details = expect_missing_artifact_error(f"cfr:{cfr_path}", cfr_path)
    check("missing CFR profile fails loudly", cfr_ok, cfr_details)

    deep_path = _missing_path(".pt")
    deep_ok, deep_details = expect_missing_artifact_error(
        f"deep_cfr:{deep_path}", deep_path
    )
    check("missing Deep CFR weights fail loudly", deep_ok, deep_details)


class _CaptureFoldBot:
    def __init__(self):
        self.views: list[PlayerView] = []

    def act(self, view: PlayerView) -> Action:
        self.views.append(view)
        return Action("fold")


class _PassiveBot:
    def act(self, view: PlayerView) -> Action:
        legal_types = {a["type"] for a in view.legal_actions}
        if "check" in legal_types:
            return Action("check")
        if "call" in legal_types:
            return Action("call")
        return Action("fold")


def test_engine_short_all_in_raise() -> None:
    section("Section 5: short all-in raise exposure")

    short_stack = _CaptureFoldBot()
    passive = _PassiveBot()
    table = Table()
    table.play_hand(
        seats=[
            Seat("BTN_SHORT", 150),
            Seat("SB", 1000),
            Seat("BB", 1000),
        ],
        small_blind=50,
        big_blind=100,
        dealer_index=0,
        bot_for={
            "BTN_SHORT": short_stack,
            "SB": passive,
            "BB": passive,
        },
        log_decisions=False,
    )

    view = short_stack.views[0]
    short_raises = [
        a for a in view.legal_actions
        if a.get("type") == "raise" and a.get("min") == a.get("max") == 150
    ]
    check(
        "engine offers below-minimum short all-in raise",
        bool(short_raises) and short_raises[0].get("all_in") is True,
        f"legal={view.legal_actions}",
    )


def test_cfr_short_all_in_callers_remain() -> None:
    section("Section 6: CFR short all-in keeps callers")

    tabular = _GameState(
        pot=200,
        stacks=[900, 900, 150],
        committed_per_seat=[100, 100, 0],
        alive=[True, True, True],
        street="preflop",
        board=[],
        hole_cards={},
        seat_order=[0, 1, 2],
        action_idx=2,
        big_blind=100,
        last_raise_size=100,
    )
    tab_next = tabular.apply_action(2, 7)
    tab_legal = tab_next.legal_actions()
    check(
        "tabular CFR does not advance while callers owe short all-in chips",
        (not tab_next.is_chance_node())
        and tab_next.to_call_for(0) == 50
        and action_types(tab_legal) == {"fold", "call"},
        (
            f"chance={tab_next.is_chance_node()} to_call0={tab_next.to_call_for(0)} "
            f"seat_order={tab_next.seat_order} legal={tab_legal}"
        ),
    )

    deep = _DeepCFRGameState(
        pot=200,
        stacks=[900, 900, 150],
        committed_per_seat=[100, 100, 0],
        alive=[True, True, True],
        street="preflop",
        board=[],
        hole_cards={},
        seat_order=[0, 1, 2],
        action_idx=2,
        history_events=[],
        deck_remaining=[],
        big_blind=100,
        last_raise_size=100,
    )
    deep_next = deep.apply_action(2, DEEP_ABSTRACT_ACTIONS.index("all_in"))
    deep_legal = deep_next.legal_actions()
    check(
        "Deep CFR does not advance while callers owe short all-in chips",
        (not deep_next.is_chance_node())
        and deep_next.to_call_for(0) == 50
        and action_types(deep_legal) == {"fold", "call"},
        (
            f"chance={deep_next.is_chance_node()} to_call0={deep_next.to_call_for(0)} "
            f"seat_order={deep_next.seat_order} legal={deep_legal}"
        ),
    )


def test_cfr_reraise_minimum_uses_last_raise() -> None:
    section("Section 7: CFR reraise minimum uses last raise size")

    tabular = _GameState(
        pot=90,
        stacks=[990, 960, 960],
        committed_per_seat=[10, 40, 40],
        alive=[True, True, True],
        street="preflop",
        board=[],
        hole_cards={},
        seat_order=[0],
        big_blind=10,
        last_raise_size=30,
    )
    tab_raise = next(a for a in tabular.legal_actions() if a.get("type") == "raise")
    check(
        "tabular CFR min reraise is current bet plus last raise size",
        tab_raise.get("min") == 70,
        f"legal={tabular.legal_actions()}",
    )

    deep = _DeepCFRGameState(
        pot=90,
        stacks=[990, 960, 960],
        committed_per_seat=[10, 40, 40],
        alive=[True, True, True],
        street="preflop",
        board=[],
        hole_cards={},
        seat_order=[0],
        action_idx=0,
        history_events=[],
        deck_remaining=[],
        big_blind=10,
        last_raise_size=30,
    )
    deep_raise = next(a for a in deep.legal_actions() if a.get("type") == "raise")
    check(
        "Deep CFR min reraise is current bet plus last raise size",
        deep_raise.get("min") == 70,
        f"legal={deep.legal_actions()}",
    )


def test_deep_cfr_search_root_preserves_bet_root() -> None:
    section("Section 8: Deep CFR search root preserves engine bet roots")

    view = PlayerView(
        me="P0",
        street="flop",
        position="BTN",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[("2", "h"), ("7", "d"), ("J", "c")],
        pot=300,
        to_call=0,
        min_raise=10,
        max_raise=1000,
        legal_actions=[
            {"type": "check"},
            {"type": "bet", "min": 10, "max": 1000},
        ],
        stacks={"P0": 1000, "P1": 1000},
        opponents=["P1"],
        history=[],
    )
    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False)
    state, hero_seat = bot._build_search_game_state(view)
    legal_types = action_types(state.legal_actions())
    check(
        "Deep CFR search root keeps to_call=0 check/bet state as check/bet",
        state.committed_per_seat == [0, 0]
        and state.to_call_for(hero_seat) == 0
        and "bet" in legal_types
        and "raise" not in legal_types,
        f"committed={state.committed_per_seat} legal={state.legal_actions()}",
    )


def test_deep_cfr_all_in_warmup_boundary() -> None:
    section("Section 9: Deep CFR schema-v2 full action exposure")

    state = _DeepCFRGameState(
        pot=15,
        stacks=[100, 90],
        committed_per_seat=[0, 10],
        alive=[True, True],
        street="preflop",
        board=[],
        hole_cards={},
        seat_order=[0],
        action_idx=0,
        history_events=[],
        deck_remaining=[],
        big_blind=10,
        ring_order=[0, 1],
    )
    labels = {
        DEEP_ABSTRACT_ACTIONS[idx]
        for idx in state.legal_abstract_actions()
    }
    check(
        "all-in is present in the legal traversal action set",
        "all_in" in labels,
        f"legal={sorted(labels)}",
    )


def test_new_table_order_and_view_contracts() -> None:
    section("Section 10: table order, seat identity, and opponent semantics")

    state = _DeepCFRGameState(
        pot=0,
        stacks=[100, 100, 100, 100],
        committed_per_seat=[0, 0, 0, 0],
        alive=[True, True, True, True],
        street="preflop",
        board=[],
        hole_cards={},
        seat_order=[3, 0, 1, 2],
        action_idx=4,
        history_events=[],
        deck_remaining=[],
        ring_order=[0, 1, 2, 3],
    )
    flop = state.advance_street()
    check(
        "Deep CFR postflop order starts left of button",
        flop.seat_order == [1, 2, 3, 0],
        f"seat_order={flop.seat_order}",
    )

    seats = [
        Seat("P0", 100),
        Seat("P1", 0),
        Seat("P2", 100),
        Seat("P3", 100),
    ]
    check(
        "dealer helper skips eliminated seats",
        normalize_dealer_seat_index(seats, 1) == 2
        and advance_dealer_seat_index(seats, 0) == 2,
        "expected inactive button at 1 and next after 0 to resolve to seat 2",
    )
    run_eval_source = inspect.getsource(run_eval._run_one_tournament)
    tournament_source = inspect.getsource(tournament_mod.run_tournament)
    check(
        "run_eval uses full-table dealer rotation",
        "run_tournament(" in run_eval_source
        and 'dealer_rotation="full_table"' in run_eval_source
        and "normalize_dealer_seat_index" in tournament_source
        and "advance_dealer_seat_index" in tournament_source
        and "dealer_index % len(active_seats)" not in run_eval_source,
        "run_eval still contains active-list dealer indexing",
    )

    class _ShortAllIn:
        def act(self, view: PlayerView) -> Action:
            return Action("raise", 150)

    class _Capture:
        def __init__(self):
            self.views: list[PlayerView] = []

        def act(self, view: PlayerView) -> Action:
            self.views.append(view)
            return Action("fold")

    class _Call:
        def act(self, view: PlayerView) -> Action:
            return Action("call")

    capture = _Capture()
    Table().play_hand(
        seats=[Seat("BTN_SHORT", 150), Seat("SB", 1000), Seat("BB", 1000)],
        small_blind=50,
        big_blind=100,
        dealer_index=0,
        bot_for={
            "BTN_SHORT": _ShortAllIn(),
            "SB": capture,
            "BB": _Call(),
        },
    )
    view = capture.views[0]
    check(
        "all-in opponent remains contesting but cannot act",
        "BTN_SHORT" in view.opponents
        and "BTN_SHORT" not in (view.acting_opponents or [])
        and "BTN_SHORT" in (view.all_in_opponents or []),
        (
            f"opponents={view.opponents} acting={view.acting_opponents} "
            f"all_in={view.all_in_opponents}"
        ),
    )


def test_new_cfr_profile_and_stats_contracts() -> None:
    section("Section 11: CFR profile lookup, stat buckets, and counters")

    base_entry = {
        "street": "preflop",
        "pid": "P2",
        "type": "raise",
        "amount": 30,
        "pot_before": 15,
    }
    view1 = PlayerView(
        me="P1",
        street="preflop",
        position="SB",
        hole_cards=[("A", "s"), ("A", "h")],
        board=[],
        pot=15,
        to_call=10,
        min_raise=20,
        max_raise=100,
        legal_actions=[{"type": "fold"}, {"type": "call"}],
        stacks={"P1": 100, "P2": 100},
        opponents=["P2"],
        history=[base_entry],
        hand_id=1,
        seat_indices={"P1": 0, "P2": 1},
    )
    view2 = PlayerView(
        me="P1",
        street="preflop",
        position="SB",
        hole_cards=[("K", "s"), ("K", "h")],
        board=[],
        pot=15,
        to_call=10,
        min_raise=20,
        max_raise=100,
        legal_actions=[{"type": "fold"}, {"type": "call"}],
        stacks={"P2": 100, "P1": 100},
        opponents=["P2"],
        history=[],
        hand_id=2,
        seat_indices={"P1": 0, "P2": 1},
    )
    bot = CFRBot(iterations=0, profile_path=None, inference_mode=True)
    with redirect_stdout(io.StringIO()):
        bot.act(view1)
        bot.act(view2)
    check(
        "opponent stats follow stable seat_indices across stack order drift",
        bot._opp_stats.stats_for(1).sample_size == 1
        and bot._opp_stats.stats_for(0).sample_size == 0,
        (
            f"seat0={bot._opp_stats.stats_for(0).sample_size} "
            f"seat1={bot._opp_stats.stats_for(1).sample_size}"
        ),
    )

    repeated = CFRBot(iterations=0, profile_path=None, inference_mode=True)
    repeated._last_history_len = 1
    repeated._last_history_snapshot = [base_entry]
    check(
        "repeated identical CFR view is not a hand boundary",
        not repeated._detect_hand_boundary(view1),
        "same hand/history was classified as a boundary",
    )

    skipped = CFRBot(iterations=3, profile_path=None, inference_mode=False)
    skipped._build_training_game_state = lambda *args, **kwargs: None
    skipped._run_iterations(
        info_key="x",
        legal_mask=[1],
        pot=15,
        hole=[("A", "s"), ("K", "s")],
        board=[],
        street="preflop",
        n_opponents=1,
        call_amount=10,
        hero_stack=100,
        view=view1,
    )
    check(
        "_run_iterations counts only completed traversals",
        skipped._total_iterations == 0,
        f"total_iterations={skipped._total_iterations}",
    )

    bare_stderr = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(bare_stderr):
        bare = create_bot("cfr")
    check(
        "bare cfr works without default artifact",
        bare is not None,
        "create_bot('cfr') raised or returned None",
    )
    # The no-crash fallback is intentional, but it must be LOUD: a blank
    # CFRBot silently poisons eval pools and RL opponent tables. Only
    # assert the warning when the default artifact is actually absent
    # (with a real profile on disk, bare cfr loads it and stays quiet).
    if not os.path.exists("models/cfr_regret_deep_v2.pkl"):
        check(
            "bare cfr fallback warns UNTRAINED on stderr",
            "UNTRAINED" in bare_stderr.getvalue(),
            f"no UNTRAINED warning on stderr; got: "
            f"{bare_stderr.getvalue()!r:.120}",
        )

    node = _CFRNode()
    node.strategy_sum[1] = 1.0
    legacy_key = "preflop:1:early:low:10:K"
    current_key = "preflop:1:early:low:LP:10:K"
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        with open(path, "wb") as fh:
            pickle.dump({"nodes": {legacy_key: node.to_dict()}}, fh)
        with redirect_stdout(io.StringIO()):
            legacy_bot = CFRBot(profile_path=path, inference_mode=True)
        check(
            "legacy CFR profile keys are usable by current lookup",
            legacy_bot._lookup_node(current_key) is not None,
            "current key did not fall back to legacy key",
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    search_bot = CFRBot(iterations=0, profile_path=None, inference_mode=True)
    search_bot._compute_opp_stat_bucket = lambda _view: "LP"
    seen: list[str] = []
    search_bot._search_subtree = (
        lambda state, hero_seat, depth: seen.append(state.opp_stat_bucket) or 0.0
    )
    with redirect_stdout(io.StringIO()):
        search_bot._subgame_search(view1, [0.0, 1.0] + [0.0] * 6, [1], depth=1)
    check(
        "CFR subgame root uses real opponent stat bucket",
        seen == ["LP"],
        f"seen={seen}",
    )

    all_in_view = PlayerView(
        me="P0",
        street="flop",
        position="BTN",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[("2", "c"), ("7", "d"), ("J", "h")],
        pot=400,
        to_call=100,
        min_raise=200,
        max_raise=1000,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 200, "max": 1000},
        ],
        stacks={"P0": 1000, "P1": 900, "P2": 0, "P3": 0, "P4": 0},
        opponents=["P1", "P2", "P3", "P4"],
        acting_opponents=["P1"],
        all_in_opponents=["P2", "P3", "P4"],
        history=[],
    )
    captured_n_opp: list[int] = []
    original_postflop_bucket = cfr_mod._postflop_bucket
    try:
        cfr_mod._postflop_bucket = (
            lambda hole, board, n_opponents: captured_n_opp.append(n_opponents) or 0
        )
        with redirect_stdout(io.StringIO()):
            CFRBot(iterations=0, profile_path=None, inference_mode=True).act(all_in_view)
    finally:
        cfr_mod._postflop_bucket = original_postflop_bucket
    check(
        "CFR info-set opponent count ignores all-in non-actors",
        captured_n_opp and captured_n_opp[0] == 1,
        f"captured_n_opp={captured_n_opp}",
    )


def test_new_deep_cfr_history_features_and_mask() -> None:
    section("Section 12: Deep CFR history and features")

    bad_history_view = PlayerView(
        me="P0",
        street="flop",
        position="BTN",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[("2", "c"), ("7", "d"), ("J", "h")],
        pot=100,
        to_call=0,
        min_raise=10,
        max_raise=100,
        legal_actions=[{"type": "check"}],
        stacks={"P0": 100, "P1": 0},
        opponents=["P1"],
        history=[{"street": "flop", "pid": "ghost", "type": "bet", "amount": 50}],
        acting_opponents=[],
        all_in_opponents=["P1"],
    )
    batch = build_network_input(bad_history_view)
    check(
        "Deep CFR network input tolerates missing history pids",
        batch["history"].shape[1] > 0,
        "build_network_input raised or returned an empty history tensor",
    )
    check(
        "Deep CFR inference opponent features use public schema",
        float(batch["opp_features"][0, 0, 0]) == 0.0
        and float(batch["opp_features"][0, 0, 2]) == 0.0
        and float(batch["opp_features"][0, 0, 3]) == 1.0,
        f"features={batch['opp_features'][0, 0].tolist()}",
    )

def test_new_chip_accounting_and_training_failure_contracts() -> None:
    section("Section 13: chip accounting and failure signaling")

    awards = _side_pot_awards(
        scores={0: 10},
        alive_tuple=(True, False, False),
        committed_per_seat=(100, 100, 200),
        pot=400,
    )
    check(
        "AIVAT side-pot orphan layers are not lost",
        math.isclose(sum(awards.values()), 400.0, abs_tol=1e-6),
        f"awards={awards}",
    )

    nine = {f"P{i}": 1000 for i in range(9)}
    eq = icm_equity(nine)
    check(
        "ICMBot ICM equity handles 9 alive players",
        abs(sum(eq.values()) - 1.0) < 1e-9 and all(v >= 0 for v in eq.values()),
        f"equities={eq}",
    )
    delta = icm_ev_of_call(
        "P0",
        {"P0": 1000, "P1": 1000, "P2": 1000},
        pot=300,
        to_call=300,
        win_prob=0.0,
        pot_recipient_pid="P1",
    )
    check(
        "ICM call EV compares against chip-conserving fold/loss states",
        delta < 0.0,
        f"delta={delta}",
    )

    class _Boom:
        def act(self, _view):
            raise RuntimeError("real bot bug")

    view = PlayerView(
        me="P0",
        street="preflop",
        position="BTN",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[],
        pot=15,
        to_call=0,
        min_raise=10,
        max_raise=100,
        legal_actions=[{"type": "check"}],
        stacks={"P0": 100, "P1": 100},
        opponents=["P1"],
        history=[],
    )
    try:
        InProcessBot(_Boom()).act(view)
    except RuntimeError:
        propagates = True
    else:
        propagates = False
    check(
        "InProcessBot propagates real bot exceptions",
        propagates,
        "RuntimeError was swallowed by dict fallback",
    )

    save_path = os.path.join(tempfile.gettempdir(), f"deep_cfr_abort_{uuid.uuid4().hex}.pt")
    args = SimpleNamespace(
        variant="small",
        iterations=0,
        round_size=25_000,
        update_interval=1,
        fit_steps=1,
        fit_batch_size=999,
        checkpoint_interval=100,
        batch_size=999,
        lr=1e-4,
        aivat_sims=1,
        curriculum_profile="sixmax",
        canary_enforce_iteration=0,
        canary_fail_patience=1,
        disable_collapse_canary=False,
        save_path=save_path,
        device="cpu",
        resume=None,
    )
    old_probe = train_deep_cfr.quick_canary_probe
    old_classify = train_deep_cfr.classify_canary
    try:
        train_deep_cfr.quick_canary_probe = (
            lambda *_args, **_kwargs: {"raw_all_in": 1.0, "search_all_in": 1.0}
        )
        train_deep_cfr.classify_canary = lambda *_args, **_kwargs: "FAIL"
        with redirect_stdout(io.StringIO()):
            result = train_deep_cfr.run_training(args)
        check(
            "final Deep CFR canary failure returns aborted status",
            result.get("status") == "aborted"
            and result.get("checkpoint_saved") is None,
            f"result={result.get('status')} checkpoint={result.get('checkpoint_saved')}",
        )
    finally:
        train_deep_cfr.quick_canary_probe = old_probe
        train_deep_cfr.classify_canary = old_classify
        try:
            os.remove(save_path)
        except OSError:
            pass


def main() -> int:
    print("sanity_review_findings.py - regression probes for review findings")
    test_matched_bet_legal_actions()
    test_tabular_inference_to_call()
    test_deep_cfr_cost_units()
    test_missing_artifact_loading()
    test_engine_short_all_in_raise()
    test_cfr_short_all_in_callers_remain()
    test_cfr_reraise_minimum_uses_last_raise()
    test_deep_cfr_search_root_preserves_bet_root()
    test_deep_cfr_all_in_warmup_boundary()
    test_new_table_order_and_view_contracts()
    test_new_cfr_profile_and_stats_contracts()
    test_new_deep_cfr_history_features_and_mask()
    test_new_chip_accounting_and_training_failure_contracts()

    print()
    print("=" * 72)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} finding regression(s) still reproduce")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("PASS: all review-finding regressions are locked and fixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

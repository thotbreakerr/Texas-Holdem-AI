"""
bots/__init__.py — Utility helpers used by runner scripts.

Provides:
  parse_players(spec_str)  -> list of (pid, btype, adapter)
  create_bot(btype)        -> BotAdapter
  escalate_blinds(...)     -> (sb, bb)
"""
import re
from core.engine import InProcessBot, RandomBot
from core.bot_api import BotAdapter, PlayerView, Action


# ── Bot creation ──────────────────────────────────────────────────────────────

def create_bot(btype: str) -> BotAdapter:
    """
    Create a bot adapter from a type string.

    Recognised types (case-insensitive):
      mc, mc<N>          MonteCarloBot (optional sim count: mc200, mc500)
      smart              SmartBot (heuristic)
      final, final_survival
                          TournamentHybridBot survival profile
      final_aggro         TournamentHybridBot aggro profile
      final_<profile>:p4|telemetry|station|r2|p5
                          TournamentHybridBot Phase 5 ablation arms
      maniac, maniac_trigger, maniac_mixed, overbet_merchant,
      calling_station, nit, folder, loose_passive, minraise, minraiser,
      baseline_sane, pressure_filler
                          Phase 7 stress-opponent archetypes
      random             RandomBot
    """
    raw_btype = btype.strip()
    btype = raw_btype.lower()

    if btype.startswith("mc"):
        from bots.monte_carlo_bot import MonteCarloBot
        m = re.match(r"mc(\d+)", btype)
        sims = int(m.group(1)) if m else 200
        return _wrap(MonteCarloBot(simulations=sims))

    if btype in ("smart", "smartbot", "heuristic"):
        from bots.poker_mind_bot import SmartBot
        return _wrap(SmartBot())

    if btype in ("random",):
        return InProcessBot(RandomBot())

    if btype in ("icm", "icmbot"):
        from bots.icm_bot import ICMBot
        return _wrap(ICMBot())

    if btype in ("exploitative", "exploitativebot"):
        from bots.exploitative_bot import ExploitativeBot
        return _wrap(ExploitativeBot())

    if btype in ("gto", "gtobot"):
        from bots.gto_bot import GTOBot
        return _wrap(GTOBot())

    if btype in ("opponentmodel", "opponentmodelbot"):
        from bots.opponent_model_bot import OpponentModelBot
        return _wrap(OpponentModelBot())

    if btype == "final" or btype.startswith(("final_survival", "final_aggro", "final:")):
        from bots.tournament_hybrid_bot import TournamentHybridBot
        profile = "aggro" if "aggro" in btype else "survival"
        bot = TournamentHybridBot(profile=profile)
        _configure_final_arm(bot, raw_btype)
        return _wrap(bot)

    if btype in (
        "maniac",
        "maniac_trigger",
        "maniac_mixed",
        "overbet_merchant",
        "calling_station",
        "nit",
        "folder",
        "loose_passive",
        "minraise",
        "minraiser",
        "baseline_sane",
        "pressure_filler",
    ):
        from bots.archetype_bot import ArchetypeBot
        return _wrap(ArchetypeBot(btype))

    raise ValueError(f"Unknown bot type: {raw_btype!r}. "
                     "Expected one of: mc, mc<N>, smart, random, "
                     "icm, exploitative, gto, opponentmodel, "
                     "final, final_survival, final_aggro, "
                     "maniac_trigger, maniac_mixed, overbet_merchant, "
                     "calling_station, nit, folder, loose_passive, minraise, "
                     "minraiser, baseline_sane, pressure_filler")


class _PlayerViewAdapter(BotAdapter):
    """Thin BotAdapter that passes PlayerView straight through."""
    def __init__(self, bot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)

    def reset_memory(self):
        """Tournament boundary: forward to bots with cross-hand state.

        Any bot with cumulative opponent memory must be
        reset when an instance is reused across Tables — a new Table
        restarts hand ids at 0, so stale dedup keys would silently swallow
        the new tournament's actions and old stats would leak in. No-op
        for bots without a reset_memory method.
        """
        reset = getattr(self.bot, "reset_memory", None)
        if callable(reset):
            reset()


def _wrap(bot) -> BotAdapter:
    """Wrap a bot object in a BotAdapter."""
    return _PlayerViewAdapter(bot)


def _configure_final_arm(bot, raw_btype: str) -> None:
    """Apply Phase 5 ablation-arm toggles to TournamentHybridBot specs."""
    spec = raw_btype.strip().lower()
    if ":" in spec:
        arm = spec.split(":", 1)[1]
    else:
        arm = spec
        for prefix in ("final_survival_", "final_aggro_", "final_"):
            if arm.startswith(prefix):
                arm = arm[len(prefix):]
                break
        if arm in ("final", "final_survival", "final_aggro"):
            arm = "p4"
    arm = arm.replace("+", "_").replace("-", "_")
    arm = {
        "": "p4",
        "survival": "p4",
        "aggro": "p4",
        "p4_telemetry": "telemetry",
        "telemetry_only": "telemetry",
        "p4_station_only": "station",
        "station_only": "station",
        "p4_strict_r2_only": "r2",
        "strict_r2_only": "r2",
        "strict_r2": "r2",
        "p4_station_r2": "p5",
        "station_r2": "p5",
        "station_strict_r2": "p5",
        "phase5": "p5",
    }.get(arm, arm)

    bot.p5_enabled = arm in {"telemetry", "station", "r2", "p5"}
    bot.p5_log_only = arm == "telemetry"
    bot.p5_station_enabled = arm in {"telemetry", "station", "p5"}
    bot.p5_r2_enabled = arm in {"telemetry", "r2", "p5"}


# ── Player-spec parsing ────────────────────────────────────────────────────────

def parse_players(spec: str):
    """
    Parse a comma-separated player spec string into a list of
    (player_id, bot_type, adapter) tuples.

    Examples:
      "mc200,smart,gto,icm"
      "P1=mc200,P2=smart,P3=gto"

    Auto-assigns P1, P2, ... when no explicit IDs are given.
    """
    entries = [s.strip() for s in spec.split(",") if s.strip()]
    result = []
    for i, entry in enumerate(entries):
        if "=" in entry:
            pid, btype = entry.split("=", 1)
            pid, btype = pid.strip(), btype.strip()
        else:
            pid = f"P{i + 1}"
            btype = entry
        adapter = create_bot(btype)
        result.append((pid, btype, adapter))
    return result


# ── Blind escalation ──────────────────────────────────────────────────────────

def escalate_blinds(hand_count: int, base_sb: int, base_bb: int,
                    blind_increase_every: int) -> tuple:
    """
    Return (sb, bb) for the given hand number.

    Blinds increase 1.5x every `blind_increase_every` hands.
    If `blind_increase_every` is 0 (or negative), no escalation occurs.
    """
    if blind_increase_every <= 0:
        return base_sb, base_bb

    level = (hand_count - 1) // blind_increase_every
    multiplier = 1.5 ** level
    sb = max(1, int(base_sb * multiplier))
    bb = max(2, int(base_bb * multiplier))
    return sb, bb

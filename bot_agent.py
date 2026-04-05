"""
Bot-Side Peak Agent

Runs inside each customer's bot. Monitors ACTUAL Binance positions,
real P&L, and real balance. Calls the signal server's /api/agent-evaluate
endpoint for AI decisions (QWEN_API_KEY never leaves the signal server).

Actions:
- CLOSE: Lock profit on a position, pause re-entry until next signal flip
- SKIP: Block a whipsawing symbol for N minutes
- HOLD: Do nothing (default)
"""

import asyncio
import logging
import os
import time
from datetime import datetime

import httpx

log = logging.getLogger("bot-agent")

SIGNAL_SERVER_URL = os.getenv("SIGNAL_SERVER_URL", "wss://signal-server-production-1802.up.railway.app")
AGENT_INTERVAL_SECONDS = int(os.getenv("AGENT_INTERVAL_SECONDS", "300"))  # 5 min
LEVERAGE = 10
MIN_PROFIT_PCT_TO_CLOSE = 2.0  # leveraged ROI % (0.2% raw price move at 10x)
COOLDOWN_SECONDS = 1800  # 30 min


class BotPeakAgent:
    """Peak agent that runs inside the customer's bot, monitoring real Binance state."""

    def __init__(self, bot):
        self.bot = bot  # ClientBot instance
        self.running = False
        self._last_action_ts: float = 0
        self._cycle_count: int = 0
        self._equity_history: list[float] = []  # rolling equity for slope calculation
        self._equity_ath: float = 0.0

    def _build_payload(self) -> dict | None:
        """Build payload from actual bot state (real positions, real prices)."""
        if not self.bot.positions or not self.bot.last_prices:
            return None

        positions = self.bot.positions
        prices = self.bot.last_prices

        # Calculate real unrealized P&L per position
        position_details = []
        unrealized_total = 0.0
        profitable_count = 0
        long_count = 0
        short_count = 0

        for symbol, pos in positions.items():
            price = prices.get(symbol, 0)
            if price <= 0:
                continue

            direction = pos["direction"]
            entry_price = pos["entry_price"]
            qty = pos.get("qty", 0)
            dm = 1 if direction == 1 else -1

            # Real P&L from actual position (leveraged ROI — matches server-side agent)
            pnl_pct = dm * (price - entry_price) / entry_price * 100 * LEVERAGE
            notional = qty * entry_price if qty else 0  # already includes leverage
            pnl_usd = dm * (price - entry_price) / entry_price * notional

            unrealized_total += pnl_usd
            if pnl_usd > 0:
                profitable_count += 1
            if direction == 1:
                long_count += 1
            else:
                short_count += 1

            # Time held
            entry_time = pos.get("entry_time", "")
            held_minutes = 0
            if entry_time:
                try:
                    held_minutes = (datetime.now() - datetime.fromisoformat(entry_time)).total_seconds() / 60
                except (ValueError, TypeError):
                    pass

            # Near flip detection from last snapshot
            near_flip = False
            # Check the bot's last received snapshot for indicator state
            if hasattr(self.bot, '_last_snapshot_symbols'):
                snap = self.bot._last_snapshot_symbols.get(f"{symbol}:USDT") or \
                       self.bot._last_snapshot_symbols.get(symbol)
                if snap:
                    don_dir = snap.get("don_dir", direction)
                    bb_dir = snap.get("bb_dir", direction)
                    near_flip = don_dir != bb_dir

            # Count recent flips for this symbol from bot's signal history
            recent_flips_30m = 0
            if hasattr(self.bot, '_signal_history'):
                thirty_min_ago = time.time() - 1800
                recent_flips_30m = sum(
                    1 for s in self.bot._signal_history
                    if s.get("symbol") == symbol and s.get("ts", 0) > thirty_min_ago
                )

            # Check skip state
            is_skipped = False
            skip_remaining = 0
            if hasattr(self.bot, '_skip_until'):
                skip_expiry = self.bot._skip_until.get(symbol, 0)
                if skip_expiry > 0 and time.time() < skip_expiry:
                    is_skipped = True
                    skip_remaining = (skip_expiry - time.time()) / 60

            position_details.append({
                "symbol": symbol,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": round(entry_price, 4),
                "current_price": round(price, 4),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "unrealized_pnl_usd": round(pnl_usd, 2),
                "held_minutes": round(held_minutes),
                "near_flip": near_flip,
                "recent_flips_30m": recent_flips_30m,
                "is_skipped": is_skipped,
                "skip_remaining_min": round(skip_remaining),
            })

        position_details.sort(key=lambda x: x["unrealized_pnl_usd"], reverse=True)

        # Estimate equity (balance + unrealized)
        # Use session_pnl as proxy for realized gains since start
        equity = self.bot.session_pnl + unrealized_total + 1000  # rough baseline

        # Try to get actual balance from last status report
        if hasattr(self.bot, '_last_balance') and self.bot._last_balance > 0:
            equity = self.bot._last_balance + unrealized_total

        # Update equity history and ATH
        self._equity_history.append(equity)
        if len(self._equity_history) > 48:  # 4 hours at 5-min intervals
            self._equity_history = self._equity_history[-48:]
        if equity > self._equity_ath:
            self._equity_ath = equity

        pct_from_ath = ((equity - self._equity_ath) / self._equity_ath * 100) if self._equity_ath > 0 else 0

        # Equity slopes
        slope_2h = 0.0
        slope_6h = 0.0
        if len(self._equity_history) >= 8:
            last_8 = self._equity_history[-8:]
            slope_2h = (last_8[-1] - last_8[0]) / abs(last_8[0]) * 100 if last_8[0] != 0 else 0
        if len(self._equity_history) >= 24:
            slope_6h = (self._equity_history[-1] - self._equity_history[0]) / abs(self._equity_history[0]) * 100 if self._equity_history[0] != 0 else 0

        cooldown_remaining = max(0, COOLDOWN_SECONDS - (time.time() - self._last_action_ts))

        return {
            "portfolio": {
                "equity": round(equity, 2),
                "unrealized_pnl": round(unrealized_total, 2),
                "equity_ath": round(self._equity_ath, 2),
                "pct_from_ath": round(pct_from_ath, 2),
                "equity_slope_2h_pct": round(slope_2h, 2),
                "equity_slope_6h_pct": round(slope_6h, 2),
                "active_positions": len(positions),
                "profitable_positions": profitable_count,
                "losing_positions": len(positions) - profitable_count,
            },
            "positions": position_details,
            "market_breadth": {
                "long_count": long_count,
                "short_count": short_count,
                "avg_unrealized_pnl_pct": round(
                    sum(p["unrealized_pnl_pct"] for p in position_details) / len(position_details), 2
                ) if position_details else 0,
            },
            "cooldown_active": cooldown_remaining > 0,
            "cooldown_remaining_minutes": round(cooldown_remaining / 60, 1),
            "cycle_number": self._cycle_count,
        }

    async def _evaluate(self, payload: dict) -> dict:
        """Call signal server's /api/agent-evaluate — Qwen key stays server-side."""
        http_url = SIGNAL_SERVER_URL.replace("wss://", "https://").replace("ws://", "http://")
        url = f"{http_url}/api/agent-evaluate"

        start = time.time()
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                url,
                params={"key": self.bot.signal_api_key},
                json=payload,
            )
            resp.raise_for_status()

        latency_ms = int((time.time() - start) * 1000)
        decision = resp.json()
        decision["latency_ms"] = latency_ms
        return decision

    async def _cleanup_orphaned_orders(self):
        """Cancel conditional orders for symbols that have no position.
        Runs each agent cycle to prevent orphan accumulation."""
        if not self.bot.proxy:
            return
        tracked_symbols = set(self.bot.positions.keys())
        cleaned = 0
        for symbol in self.bot.active_symbols:
            if symbol not in tracked_symbols:
                try:
                    await self.bot.proxy.cancel_all_orders(symbol)
                    cleaned += 1
                except Exception:
                    pass
        if cleaned > 0:
            log.info(f"Bot agent: cleaned orphaned orders for {cleaned} untracked symbols")

    async def _execute_decision(self, decision: dict, payload: dict):
        """Execute agent decisions on actual Binance positions."""
        actions = decision.get("actions", [])
        if not actions:
            log.info(f"Bot agent #{self._cycle_count}: HOLD — {decision.get('reasoning', '')[:100]}")
            return

        closed = []
        skipped = []

        for action in actions:
            symbol = action.get("symbol", "")
            action_type = action.get("action", "")

            if action_type == "CLOSE":
                # Validate profit threshold
                pos_data = next((p for p in payload["positions"] if p["symbol"] == symbol), None)
                if pos_data and pos_data["unrealized_pnl_pct"] < MIN_PROFIT_PCT_TO_CLOSE:
                    log.warning(f"Bot agent: skipping CLOSE {symbol} — only {pos_data['unrealized_pnl_pct']:.1f}%")
                    continue

                pos = self.bot.positions.get(symbol)
                if pos:
                    log.info(f"Bot agent: CLOSING {symbol}")
                    close_price = self.bot.last_prices.get(symbol, pos["entry_price"])
                    await self.bot._close_position(symbol, close_price, "bot_agent_peak")
                    # Block re-entry until next signal flip
                    if not hasattr(self.bot, "_agent_paused"):
                        self.bot._agent_paused = set()
                    self.bot._agent_paused.add(symbol)
                    closed.append({
                        "symbol": symbol,
                        "action": "CLOSE",
                        "reason": action.get("reason", "peak detected"),
                    })

            elif action_type == "SKIP":
                duration = int(action.get("duration_minutes", 30))
                pos = self.bot.positions.get(symbol)
                if pos:
                    log.info(f"Bot agent: SKIPPING {symbol} for {duration}m")
                    close_price = self.bot.last_prices.get(symbol, pos["entry_price"])
                    await self.bot._close_position(symbol, close_price, "bot_agent_skip")
                # Block re-entry (time-based)
                if not hasattr(self.bot, "_skip_until"):
                    self.bot._skip_until = {}
                self.bot._skip_until[symbol] = time.time() + duration * 60
                skipped.append({
                    "symbol": symbol,
                    "action": "SKIP",
                    "duration_minutes": duration,
                    "reason": action.get("reason", "whipsaw"),
                })

        all_actions = closed + skipped

        if all_actions:
            self._last_action_ts = time.time()

        if closed:
            symbols_str = ", ".join(c["symbol"] for c in closed)
            log.info(f"Bot agent #{self._cycle_count}: CLOSED {len(closed)} ({symbols_str})")

        if skipped:
            symbols_str = ", ".join(f"{s['symbol']}({s['duration_minutes']}m)" for s in skipped)
            log.info(f"Bot agent #{self._cycle_count}: SKIPPED {len(skipped)} ({symbols_str})")

        # Telegram notification
        if all_actions and self.bot.alerts.enabled:
            action_lines = []
            for a in all_actions:
                action_lines.append(f"  {a['action']}: {a['symbol']} — {a.get('reason', '')}")
            msg = (
                f"🤖 <b>AI Agent Action</b>\n"
                f"{chr(10).join(action_lines)}\n\n"
                f"<i>{decision.get('reasoning', '')[:200]}</i>"
            )
            await self.bot.alerts._send(msg)

    async def run(self):
        """Main agent loop — runs every 5 minutes inside the bot.
        Calls signal server /api/agent-evaluate — no API keys needed locally."""
        self.running = True
        log.info(f"Bot agent started — interval={AGENT_INTERVAL_SECONDS}s, server={SIGNAL_SERVER_URL}")

        # Wait for bot to initialize and get first prices
        await asyncio.sleep(90)

        while self.running and self.bot.running:
            self._cycle_count += 1
            try:
                payload = self._build_payload()
                if payload is None:
                    log.debug("Bot agent: no payload (bot not ready)")
                    await asyncio.sleep(AGENT_INTERVAL_SECONDS)
                    continue

                if payload["cooldown_active"]:
                    log.debug(f"Bot agent: cooldown ({payload['cooldown_remaining_minutes']:.0f}m)")
                    await asyncio.sleep(AGENT_INTERVAL_SECONDS)
                    continue

                # Cleanup orphaned orders each cycle
                await self._cleanup_orphaned_orders()

                decision = await self._evaluate(payload)
                await self._execute_decision(decision, payload)

            except httpx.HTTPStatusError as e:
                log.error(f"Bot agent API error: {e.response.status_code}")
            except Exception as e:
                log.error(f"Bot agent error: {e}")

            await asyncio.sleep(AGENT_INTERVAL_SECONDS)

    def stop(self):
        self.running = False

# Real-time BTC research collector (read-only, optional BRTI)

This is an **additional data collector** on top of the existing `--predict`
monitor. It does **not** place orders, optimize live decisions, or provide
profitable trading signals. It prepares timestamped data for future ML.

## What it collects

- **No API credentials needed (default):** Coinbase BTC-USD public ticker
  WebSocket (independent exchange tick source times and local receipt times),
  plus Kalshi production REST best bid/ask, displayed executable depth, strike,
  close time and volume every five seconds. These are **two different feed
  clocks**; use the local reception times for conservative alignment.
- **Optional Kalshi WebSocket (auth required):** Kalshi YES/NO *resting bid*
  snapshot and incremental delta messages, publicly executed trade
  notifications, plus the **CF Benchmarks BRTI** reference value if that
  subscription is enabled for your account. Source server timestamps and
  receiver timestamps are both persisted. The order book is invalidated and
  resubscribed if incremental sequence numbers do not match.
- **Separate SQLite:** `data/research_stream.db`, with
  `research_prices`, `research_quotes`, `research_markets` and
  `research_events`. No secrets, order IDs, account data, or transactions.

**Important:** The Kalshi WebSocket handshake *always* requires an API key;
we do not attempt to bypass it. The collector does not automatically reuse
your normal Kalshi order-trading key. Do not grant it an unrestricted
production key. The REST mode is already useful without one. Coinbase prices
are a proxy, not BRTI.

## SSH setup

First checkout the new development branch without overwriting existing data:

```bash
cd ~/Documents/kalshi-btc-15m-bot
git fetch origin
git switch feature/realtime-research-collector
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check .
```

In a separate SSH terminal or tmux window, run the new **keyless** collector:

```bash
cd ~/Documents/kalshi-btc-15m-bot
source .venv/bin/activate
mkdir -p logs
python -m src.research_stream 2>&1 | tee -a logs/research_stream.log
```

Keep your existing `python -m src.main --predict` process running (in a
different terminal) to continue producing the current signal audit and
periodically backfilling settled outcomes. The new research collector makes
no changes to its execution logic.

In a new SSH session:

```bash
cd ~/Documents/kalshi-btc-15m-bot
source .venv/bin/activate
python -m src.research_stream --status
python -m src.research_stream --backup
```

Backups use SQLite's online backup API and go to `data/backups`. The older
`python -m src.backup` backs up the **original** bot DB, not this one.

The collector will keep running only while the foreground SSH session
remains alive; tmux or a separate launchd service can make it persistent.
Avoid starting two research collectors against the same database.

## Optional authenticated Kalshi market-data WebSocket

**Only** if you already have a separately managed, read-only production
Kalshi API credential, set in your local `.env` (never commit):

```ini
RESEARCH_KALSHI_API_KEY_ID=YOUR_READ_ONLY_KEY_ID
RESEARCH_KALSHI_PRIVATE_KEY_PATH=./secrets/read_only_research.pem
```

Then run:

```bash
python -m src.research_stream --kalshi-ws
```

This signed connection subscribes exclusively to
`orderbook_delta`, `trade`, and `cfbenchmarks_value` for BRTI;
it uses **no order, portfolio, or balance endpoints**. RSA and Ed25519
private keys are supported. A separate production API key may have
permissions you do not want this process to hold; use the keyless mode if
in doubt. A valid WebSocket login also does not guarantee BRTI entitlement.
Unavailable subscriptions are recorded as errors rather than invented ticks.

## Future reversal labels: offline, after settlement

Once several **complete and settled** 15-minute BTC markets are recorded:

```bash
python -m src.research_labels \
  --research-db data/research_stream.db \
  --outcomes-db data/kalshi_btc15m.db \
  --csv data/research/reversal_labels.csv \
  --target 0.10 --stop 0.10 --fee 0.02 --slippage 0.01
```

This exports *one potential paper entry every 30 seconds per market and
side*, only where at least one contract of order-book depth is available.
Features use BTC/BRTI ticks timestamped **at or before** each entry.
The future label is the **first** subsequent hypothetical take-profit or
stop-loss quote, or `no_trigger`. Missing last-minute or gap-filled price
history becomes `incomplete_quotes` (not a false negative). All outcomes
are matched only after the market settles.

- Bid/ask are used rather than Kalshi's last-trade price.
- Two fee and slippage assumptions are applied to hypothetical exits.
- The source of each quote is recorded; streaming and REST are not spliced
  together inside a replay.
- `complete_quote_coverage=1` indicates only observation continuity,
  **not** a fill, a guaranteed exit, or a reliable latency-arbitrage edge.
- Group observations by `ticker` and split **chronologically** when
  eventually training ML. These 30-second examples are not independent
  bets; do not random-split records from the same market.

The fees used here are **explicit scenarios**, not a live Kalshi fee
schedule. Slippage and available depth cannot prove a live fill. Settlement
results are based on the existing bot's public outcome backfill.

## Operational limits

Coinbase websocket frames may arrive late or out of order, and Kalshi REST
responses are not synchronized atomically with BTC ticks. Kalshi WebSocket
uses ordered deltas but can reconnect on gaps. We log the source/receipt
times and drop invalid order books, rather than invent precise arbitrage
timing. The dedicated research database can grow substantially; inspect
disk use and take regular backups.

Data docs: [Coinbase Exchange WebSocket](https://docs.cdp.coinbase.com/exchange/websocket-feed/overview),
[Kalshi WebSocket authentication](https://docs.kalshi.com/getting_started/quick_start_websockets),
[Kalshi orderbook deltas](https://docs.kalshi.com/websockets/orderbook-updates),
[CF Benchmarks BRTI feed](https://docs.kalshi.com/websockets/cfbenchmarks-value).

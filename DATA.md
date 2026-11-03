# Dataset: US stocks, daily, 2016 to the latest close

Free, automated data for backtesting a US momentum strategy: daily prices from Alpaca and
point-in-time shares outstanding and sector from SEC EDGAR, joined into one daily panel.

```python
from stockdata import load_panel
df = load_panel("2016-01-01", "2026-12-31", universe_only=True)
```

## Sources

| What | Source | Notes |
|---|---|---|
| Security list | Alpaca `GET /v2/assets?status=active&asset_class=us_equity` (paper-api) | currently listed only (survivorship bias accepted) |
| Daily bars | Alpaca `GET /v2/stocks/bars`, `timeframe=1Day`, `feed=sip` | 2016-01-01 onward; IEX only if SIP is refused (logged in `data/prices/meta.json`) |
| Ticker to CIK | `www.sec.gov/files/company_tickers_exchange.json` | `BRK.B` is looked up as `BRK-B` |
| Company metadata, SIC | `submissions.zip` (EDGAR bulk, nightly) | filer type, latest annual form, fund forms |
| Share counts | `companyfacts.zip` (EDGAR bulk, nightly) | company totals (facts without a class breakdown) |
| Share counts by class | Financial Statement Data Sets `YYYYqN.zip` | only to total up multi-class filers (Visa, Berkshire, Meta...) |
| ADS ratio | latest 20-F text (EDGAR archive) | ordinary shares per ADS, cached per filing |

Credentials come from env vars `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` (headers
`APCA-API-KEY-ID` / `APCA-API-SECRET-KEY`) and `SEC_USER_AGENT`. They are never logged.
Requests are rate limited (Alpaca 185/min of the 200/min free limit, SEC 8/s of 10/s) and
retried with backoff on 429/5xx/network errors.

## Pipeline

1. **Security master** (`stockdata/universe.py`, `adr.py`). Alpaca assets on NYSE, Nasdaq, NYSE
   American, NYSE Arca, Cboe BZX and TXSE that are tradable. Each asset gets a `security_type`:
   `common`, `adr`, or a drop reason: `etf_fund`, `preferred`, `warrant`, `unit`, `right`, `debt`,
   `partnership`, `trust_units`, `otc`, `not_tradable`, `when_issued`. Rules use the symbol suffix
   (`.PRx`, `.WS`, `.U`, `.RT`), the name (ETF/ETN/Fund/Warrant/Preferred/Notes/L.P./Units...), SEC
   data (SIC 6221 commodity pools, filers of N-CSR/N-PORT with no 10-K/10-Q) and Nasdaq fifth-letter
   codes. Business development companies (file 10-K, no SIC) stay `common`. Every drop is listed in
   `data/universe/dropped.csv` with its reason.
2. **Prices** (`stockdata/prices.py`). Two passes per ticker: `adjustment=all` (open/high/low/close)
   and `adjustment=raw` (close, volume, VWAP, trades). Requests carry about 100 symbols and follow
   `next_page_token`. The split factor is `adjusted volume / raw volume`: Alpaca scales volume for
   splits only, so this is exactly the cumulative split ratio (AAPL before 2020-08-31 = 4.0).
3. **Shares** (`stockdata/shares.py`). One count per filing, in order of preference:
   `dei:EntityCommonStockSharesOutstanding` (cover page), `us-gaap:CommonStockSharesOutstanding` (or
   IFRS `NumberOfSharesOutstanding`), weighted-average basic, weighted-average diluted. The cover value
   wins unless two other counts from the same filing agree with each other and the cover is more than
   3x below them (BABA's 2026 20-F cover said 1.86bn, its balance sheet and weighted average 18.6bn) or
   about 1,000x above them with an implausible count (units error). A cover far above the period-end
   counts is kept: stock issued in a merger or SPAC closing after quarter end makes it legitimately
   larger. A single disagreeing balance-sheet count never overrides the cover: many filers tag issued
   shares (incl. treasury) as outstanding (BKNG, ZBH) or report in thousands (CAL); references about
   1,000x or 1,000,000x below the cover are rescaled when the result is a plausible count. Without a
   cover value, the weighted average beats a balance-sheet count that disagrees with it by more than
   15%. Values under 10,000 shares are ignored. Every override is in
   `data/reports/cover_values_rejected.csv`. Filers that report only per class get the classes added up
   from the Financial Statement Data Sets. Isolated outliers (a >2.5x jump that reverts at the next
   filing) are dropped and listed in `data/reports/share_problems.csv`.
4. **Panel** (`stockdata/panel.py`). Prices joined with the latest share count that was usable on
   each date, market cap, sector and the universe flag. Before that, each ticker's history is trimmed
   to start after its last stretch of 20+ sessions without trading: Alpaca returns everything ever
   traded under a symbol string, so a reused ticker (today's POM vs Pepco Holdings in 2016) or a
   successor (Dow Inc. vs Dow Chemical, carried forward at zero volume from 2017-09 to 2019-04) would
   otherwise bring another company's prices. Trims are listed in `data/reports/history_trims.csv`.
5. **Validation** (`stockdata/validate.py`) writes `data/reports/validation.md`.

## Point-in-time rules

- A filing is usable from the **first trading session after its filing date** (`FILING_LAG_TRADING_DAYS = 1`).
  EDGAR filing dates do not tell whether a filing came before or after the close, so this guarantees
  no filing is used before the market could see it. Tests check that `shares_filed < date` on every row.
- A later filing that reports an older as-of date (e.g. an amendment) never replaces a newer count.
- A count older than 550 days is treated as unknown (no market cap).
- **Split basis.** A cover-page count is as of its cover date. Balance-sheet and weighted counts are
  restated for splits up to the filing date (US GAAP restates share data retroactively for splits made
  before the statements are issued), so their basis date is the filing date. Each count is converted to
  today's split basis with the primary ticker's split factor on its basis date, and back to each day's
  basis with that day's split factor:
  `mcap(t) = close_raw(t) / split_factor(t) * shares_reported * split_factor(basis_date)`.
  This is recomputed on every update, so a new split never breaks older history.
- **Reorganized companies.** When a holding-company reorganization or redomicile moved the ticker to a
  new CIK (XOM 2026, Bunge 2023, AngloGold 2023, BlackRock 2024, Navient 2026...), the new CIK inherits
  the old CIK's filings from before its own first filing (`stockdata/successors.py`, links in
  `data/reports/successor_links.csv`, with the evidence used). The old CIK must have no ticker today; its
  name without legal words (Inc, Corp, Ltd, plc, Holdings, Group...) must be a prefix of the new name or
  a former name, or it must be a SPAC that filed jointly with the new CIK; it must hand over within 400
  days with a share count within 2x; and its own reported public float must fit its share count times
  this ticker's price (15%-160%), or, without float data, the names must be identical. This rejects
  look-alikes such as a shell called "Square Chain Corp" for Block.
- **Cover dates after the filing date** (tagging errors, up to 10 years off) are clamped to the filing date.
- **Multi-class companies.** Shares are company totals (all classes). One share class per CIK is
  `is_primary` (highest median dollar volume). `mcap` is the company market cap (total shares x primary
  class price) on every class's rows. Only the primary class can be `in_universe`, so GOOGL and GOOG
  are never both selected. Berkshire's counts are in Class B equivalents (its 2016-2017 filings only give
  Class A equivalents, converted at the fixed 1 A = 1,500 B), so BRK.B is primary.
- **ADRs.** 20-F filers report ordinary shares; for ADRs the count is divided by the ADS ratio
  (ordinary shares per ADS) parsed from the Alpaca name or the latest 20-F. Earlier ratios are inferred:
  a jump in ordinary shares by a clean factor with no ADS split means a home-market split changed the
  ratio (BABA 2019: 8-for-1, 1 ADS = 1 -> 8 shares). ADRs keep prices and a best-effort `mcap` but are
  excluded from `in_universe` by default (`UNIVERSE_SECURITY_TYPES`).

## Sector

SIC (current value from `submissions`) is mapped to 11 GICS-style sectors in `stockdata/sectors.py`:
Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials,
Information Technology, Communication Services, Utilities, Real Estate. A few mega-caps whose SIC
contradicts GICS are overridden by CIK (Alphabet, Meta: Communication Services; Visa, Mastercard,
PayPal, Fiserv, FIS, Global Payments: Financials). Operating filers with no SIC (business development
companies) map to Financials.

## Schema

### `load_panel()` (`data/panel/panel.parquet`, sorted by date, ticker)

| column | meaning |
|---|---|
| `date` | session date (New York) |
| `ticker` | Alpaca symbol (class shares as `BRK.B`) |
| `open, high, low, close` | fully adjusted (splits, dividends, spin-offs) as of the last update: use for signals |
| `close_raw` | unadjusted close as traded that day: use for the price filter |
| `volume` | unadjusted shares traded |
| `dollar_volume` | `close_raw * volume` |
| `adv50` | mean `dollar_volume` over the last 50 sessions (empty until 50 sessions exist) |
| `split_factor` | cumulative split factor after this day; `close_raw / split_factor` = split-adjusted price |
| `shares` | shares outstanding known on that date, in this ticker's as-traded units (`mcap / close_raw`) |
| `shares_asof` | date the share count refers to |
| `shares_filed` | filing date of that count (always before `date`) |
| `shares_source` | `kind:origin`, kind = `eso` cover, `cso` balance sheet, `wab`/`wad` weighted average |
| `mcap` | company market cap in USD, point-in-time |
| `sector` | GICS-style sector |
| `cik` | SEC CIK |
| `security_type` | `common` or `adr` |
| `is_primary` | the share class used for the company |
| `data_issue` | set when prices look broken (see below); such rows are never `in_universe` |
| `in_universe` | common, primary, `mcap >= 3e9`, `close_raw >= 10`, `adv50 >= 25e6`, no `data_issue` |

Other tables: `load_securities()` (security master incl. dropped assets), `load_prices()` (price store),
`load_share_events()` (one row per filing used, with `filed`, `effective`, `shares_reported`, `shares_adj`).

## Refreshing

```bash
python -m stockdata update          # incremental: new sessions, re-adjusted tickers, SEC if older than 20h
python -m stockdata update --full   # re-download all prices
python -m stockdata rebuild         # recompute shares/panel/reports from local files
python -m stockdata status          # freshness, timings, background sync state
python -m pytest                    # tests (loader + point-in-time rules; real-data checks if built)
```

Incremental price logic: each run re-downloads the last 5 sessions for every ticker. If Alpaca's
adjusted closes for those days changed (a new dividend or split re-adjusts all history), that ticker's
adjusted history is downloaded again; raw history is only appended. New tickers get full history.

## Persistence: new cloud sessions

Every cloud session starts on a fresh machine with no `data/`. Measured here (4 vCPU, 16 GB):

| step | time |
|---|---|
| full rebuild from nothing (`python -m stockdata update` on an empty `data/`) | FULL_REBUILD |
| of which: prices, 10.5M daily bars x 2 passes at the free 200 req/min | ~13 min |
| of which: SEC downloads (3 GB bulk + 51 quarterly data sets) + ADR checks, in parallel with prices | ~9 min |
| of which: share extraction, panel, validation (after prices) | ~4 min |
| incremental top-up, one new session, SEC not due | ~3 min |
| restore an encrypted snapshot (download, decrypt, unpack) | RESTORE_TIME |

Chosen approach (simplest that is fast and keeps market data private):

1. **SessionStart hook** (`.claude/hooks/session-start.sh`, registered in `.claude/settings.json`). It
   installs dependencies when they change and starts `python -m stockdata sync` fully detached, so the
   session is never held up. `sync` restores the latest snapshot when `STOCKDATA_SNAPSHOT_KEY` is set,
   otherwise builds from scratch, then tops up incrementally. Progress: `python -m stockdata status`,
   log in `data/logs/sync.log`. Re-runs on resume/compact are no-ops (lock + 6-hour freshness check).
2. **Encrypted snapshot on a GitHub release**, built by `.github/workflows/data-snapshot.yml` (weekdays
   23:37 UTC and on demand). The repository is public and Alpaca market data may not be redistributed,
   so the snapshot is always encrypted (AES-256, openssl, key `STOCKDATA_SNAPSHOT_KEY`); the pack step
   refuses to run without the key. The snapshot excludes the 3 GB SEC bulk zips (re-downloaded only when
   SEC data is refreshed, at most once per 20 hours). Size: SNAPSHOT_SIZE.

Without the snapshot (nothing configured) sessions still work: the hook rebuilds in the background and
data is ready after about FULL_REBUILD. With the snapshot, data is ready in about RESTORE_TIME.

To enable the snapshot (one-time):
- GitHub repo, Settings > Secrets and variables > Actions: add `ALPACA_API_KEY_ID`,
  `ALPACA_API_SECRET_KEY`, `SEC_USER_AGENT` and `STOCKDATA_SNAPSHOT_KEY` (any long random string).
- Claude Code cloud environment settings: add `STOCKDATA_SNAPSHOT_KEY` with the same value.
- Run the `data-snapshot` workflow once (Actions tab, Run workflow), or wait for the schedule.

## Known gaps

See `data/reports/validation.md` and `data/reports/problems.csv` for the current list.

- Survivorship bias: only currently listed stocks (by design, for now). Tickers that later delist stay
  in the price store with their history.
- Sector is today's SIC, not point-in-time, and SIC is an approximation of GICS.
- Banks that file with the FDIC/OCC instead of the SEC (e.g. TowneBank, Hingham) have no CIK, so no
  shares or market cap.
- Multi-class companies that report only per class rely on the quarterly Financial Statement Data Sets,
  which are published after each quarter ends, so their newest filing can lag by up to about 3 months.
  Visa's classes are added unconverted (B/C shares count 1:1), slightly understating its market cap.
- ADRs: the ADS ratio is today's; a past ratio change that Alpaca did not treat as a split distorts older
  ADR market caps. ADRs are outside the default universe.
- Unadjusted price jumps: a one-day raw price ratio within 3% of a clean fraction (x2, x0.5, x5...)
  with no recorded split, the volume level moving by the inverse ratio and no event-day volume spike is
  treated as a corporate action Alpaca did not adjust. The next 260 sessions get `data_issue` and leave
  the universe, because any 12-month lookback spanning the jump is wrong. Found: AZN 2026-02-02 (ADRs
  replaced by ordinary shares, 1 ADS = 0.5 share), IX 2025-02-28 (ADS ratio change), SLGN 2017-05-30
  (2-for-1 split not applied), ZWS 2021-10-05 (Rexnord/Regal transaction), VEL 2020-03-16. Real crashes
  come with a volume spike and are not flagged, so losers are not silently removed.
  Listed in `data/reports/price_jumps.csv`.
- Alpaca's multi-symbol paging once skipped a symbol entirely (DOW); symbols that come back empty from a
  batch are re-requested one by one.
- Shares come from XBRL filings, so a company's first value appears with its first 10-Q/10-K/20-F after
  listing; recent IPOs have no market cap until then.
- Business development companies (ARCC, BXSL, OBDC, FSK, MAIN...) only started tagging XBRL in 2022, so
  they have no market cap, and cannot be in the universe, before their first tagged filing.
- Tracking stocks (Liberty Media's FWONA/FWONK and similar) share one CIK with other trackers, so their
  "company" share count is not specific to the tracker; check `data/reports/problems.csv`.

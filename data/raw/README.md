# Real-Stream Datasets (Section 6.8)

Two datasets from the **Numenta Anomaly Benchmark** (NAB), MIT licence,
redistributed here for one-step reproducibility of Section 6.8.

| File | Records | Resolution | Period | Source |
|---|---|---|---|---|
| `nyc_taxi.csv` | 10,320 | 30 min | 2014-07-01 to 2015-01-31 | NAB `realKnownCause/nyc_taxi.csv` |
| `nab_machine_temperature.csv` | 22,695 | 5 min | Industrial sensor preceding documented system failure | NAB `realKnownCause/machine_temperature_system_failure.csv` |

## Format

Both files share the same NAB CSV schema:

```
timestamp,value
2014-07-01 00:00:00,10844
2014-07-01 00:30:00,8127
...
```

- `timestamp` — ISO 8601 datetime.
- `value` — continuous numeric reading (taxi pickup count or temperature).

## Why these two

- **NYC Taxi** combines daily and weekly seasonality with documented burst
  events (NYC Marathon, Thanksgiving, snowstorms). Used in the paper as the
  *favourable-region* case: empirical $\tau = 6$, $r = 2.77$, proxy $\approx
  0.88$.
- **Machine Temperature** combines slow drift over hours-to-days with
  localised spike events. Used as the *unfavourable-region* case: empirical
  $\tau = 32$, $r = 13.55$, proxy saturated at $0.50$.

The pair was chosen *before* examining the results to span very different
correlation regimes; we pre-committed to reporting whichever placement and
baseline outcomes emerged.

## Upstream

Original repository: <https://github.com/numenta/NAB> (MIT licence).

To re-download from upstream:

```bash
curl -L -o nyc_taxi.csv \
  https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
curl -L -o nab_machine_temperature.csv \
  https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv
```

## Citation

> Ahmad, S., Lavin, A., Purdy, S., Agha, Z. (2017). *Unsupervised real-time
> anomaly detection for streaming data.* Neurocomputing 262, 134–147.
> [doi:10.1016/j.neucom.2017.04.070](https://doi.org/10.1016/j.neucom.2017.04.070)

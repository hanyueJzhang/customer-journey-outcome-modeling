# Data Handling Policy

The original capstone data is not included because it contains high-volume, event-level customer journey records and identifiers that are not appropriate for a public portfolio repository.

## Accepted Schema

Use an authorized table with these columns:

| Column | Type | Meaning |
|---|---|---|
| `journey_id` | string | Pseudonymous journey identifier |
| `event_name` | string | Normalized behavioral event token |
| `event_timestamp` | datetime | Timestamp for event ordering and snapshot construction |

## Required Controls

1. Remove customer, account, email, phone, address, payment, and order identifiers upstream.
2. Replace any remaining join key with a scoped pseudonymous journey key.
3. Keep raw and derived row-level data outside version control.
4. Verify that event names do not contain free text or embedded identifiers.
5. Document the source, license, retention period, and permitted use of replacement data.
6. Review aggregates before publication when small groups could enable re-identification.

Run `examples/run_synthetic_demo.py` to exercise the pipeline without external data.


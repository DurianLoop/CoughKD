# Data Access Tracker Usage

This directory tracks external datasets needed for CoughKD-ShiftAudit.

Files:

- `dataset_access_tracker.csv`: status table for external targets.
- `data_access_action_pack.md`: detailed rationale, links, templates, and commands.
- `emails/cambridge_covid19_sounds_request.txt`: ready-to-send Cambridge request draft.
- `emails/dicova_track1_request.txt`: ready-to-send DiCOVA request draft.

Suggested workflow:

1. Send the Cambridge email first.
2. Start CODA TB/Synapse access in parallel if Cambridge response is slow.
3. Send DiCOVA request only as auxiliary protocol access.
4. Update `dataset_access_tracker.csv` status values manually: `not_requested`, `requested`, `approved`, `downloaded`, `manifest_ready`, `evaluated`, or `blocked`.
5. After any manifest is ready, run `scripts/onboard_external_target.py`.

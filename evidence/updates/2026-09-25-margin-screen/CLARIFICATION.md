# 2026-09-25: diagnosis audit terminology

The copied lesson's phrase "20 logged training checkpoints" refers to twenty
logged training-step records: ten metric/history rows per arm at steps 200,
400, ..., 2000. It does not mean twenty saved model checkpoint artifacts or
twenty tensor replays. The audit field is `training_logged_steps_checked=20`.

The live source lesson and paper 8 now use the precise wording. The dated
evidence copy remains unchanged, with this clarification appended to its
checksum manifest. No measured value, diagnosis or decision changes.

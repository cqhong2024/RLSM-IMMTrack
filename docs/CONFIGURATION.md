# Frozen configuration

The public runner defaults to configs/paper.json. These values reproduce the archived configuration; they were not selected on a new validation split during packaging.

| JSON key | Default | Meaning |
|---|---:|---|
| seed | 20260817 | NumPy/PyTorch initialization seed |
| disable_camera | false | Disable inherited camera-motion estimation when true |
| disable_redetection | false | Disable inherited re-detection proposals when true |
| fixed_measurement_noise | false | Use fixed instead of confidence-controlled measurement covariance when true |
| short_capacity | 6 | Maximum short-term appearance entries |
| long_capacity | 4 | Maximum long-term entries, including protected initial reference |
| memory_supports | 3 | Subsequent supportive observations needed before commitment |
| long_weight | 0.70 | Long-bank weight in the combined appearance score |
| disagreement_penalty | 0.25 | Penalty for long-/short-bank score disagreement |
| entry_streak | 3 | Consecutive strong wins required for shadow takeover |
| entry_evidence | 0.045 | Minimum aggregated evidence for a strong win |
| exit_patience | 2 | Unsuccessful active-shadow steps tolerated before rollback |

Additional acceptance margins, noise schedules, model transitions and safeguards remain in the preserved algorithm code. This is not a claim that the JSON exposes every numerical constant. Modifying a safeguard changes the method and invalidates exact paper reproduction.

Use a new config file and output directory for every ablation. The manifest records the full resolved configuration and fingerprints of the algorithm and vendored model source. Do not select parameters on a held-out test set and then report that set as independent validation.

## Causal state ownership

1. Initialize the main SUTrack templates, IMM and shadow anchor using only the first GT box.
2. At each frame, run the safe closed-loop branch; preserve premeasurement templates for the shadow probe.
3. Predict the shadow center using its previous velocity, holding its previous width and height before measurement.
4. Compare the resulting shadow candidate with the current safe output and apply the consecutive-win/rollback gates.
5. Return the chosen box, but retain the safe branch's tracker state, IMM posterior and updated deep templates.
6. Validate appearance proposals against later safe-branch observations. On rollback, quarantine any pending proposal.

The implementation of these transitions is in RiskLimitedShadowTracker, including _shadow_prediction, _advance_shadow and _reset_shadow. The two branches are logically isolated within a single process, not two independently trained networks.

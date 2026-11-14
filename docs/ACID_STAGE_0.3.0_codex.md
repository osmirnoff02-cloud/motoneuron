# 0.3.0: optional PMCA proton / Na-H exchange closure

Codex, 2026-11-14. Original acid block by Claude; integrated and audited by Codex.
This is a conditional research prototype, not a calibrated healthy human neuron.

Completed computational substage: N05.1.4 connection; N04.2.3 recovery screen;
N17.1.3 initial historical regression and post-load Con2 probes. These parent
tasks remain partial: waveform, pH feedback and human calibration are open.

## Run

Use the existing installation instructions. Acid removal is OFF by default.
Choose the scenario explicitly; retain identical options when restarting:

```powershell
python -B run_demo_codex.py --protocol smoke --profile Selvaraj-Con2 --acid-tau-s 120 --check-restart --output outputs/acid-smoke
python -B run_demo_codex.py --protocol pulse --profile Burley --current-pa 25 --acid-tau-s 120 --output outputs/acid-pulse
python -B run_demo_codex.py --protocol rest72 --profile Selvaraj-Con2 --acid-tau-s 120 --output outputs/acid-rest
python -B run_demo_codex.py --protocol load36 --profile Selvaraj-Con2 --acid-tau-s 120 --branch repeated --checkpoint outputs/acid-rest/checkpoint_codex.json --output outputs/acid-repeated
```

Also run an independent `--branch control` from the same complete rest checkpoint.
No state is reset between the six one-hour AMPA intervals; 36-hour controls and
loads follow 72 hours of preparation. Equal opening duration is not equal Ca dose.
Long protocols are computationally expensive and were checked only for Con2.

## Results and limits

25 main calculations passed their finite-state and material/charge criteria.
The initial 15/25/50 pA counts remain Burley 3/4/3, Con1 5/6/4, Con2 5/6/7.
These are historical regression targets, not nine independent human measurements.
Selvaraj 50 pA is not a published experimental target.

The repeated-load residual process K fell from about -1202 to -1.22 uM relative
to a matched-age control. All three exchange times (20/120/600 s) improved recovery.
This does not identify a human recovery time. Minimum saved process ATP was
99.973% and phosphocreatine 98.720% of start at tau=120 s; the previous C08 also
had no severe energy collapse. Long trajectories were sampled every 300 s.

Na/H exchange carries no direct current or ATP debit. The existing Na/K pump
handles the extra Na and its ATP cost. Proton equivalents and cumulative ledgers
add six states, giving 4431 with the chloride block. H/Ca is fixed at two;
zero is only an engineering control. Buffer and resting pH affect an observer,
not channel or enzyme kinetics. CK/AK retain the previous pH 7.0 passport.
No bicarbonate route, extracellular proton pool, terminal acid block or complete
metabolic acid sources are present. Basal Na/H is assumed inside the fitted Na
background. Molecular TDP-43 drift near 2% over 72 hours remains. Recordings still
use historical 23 C / Na 145 mM conditions, not all publication-specific baths.

See [numerical acceptance](../results/acid_acceptance_results_codex.json),
[electrical audit](../results/acid_electrical_audit_codex.json),
[scenario ranges](../parameters/acid_extrusion_codex.json) and
[recovery figure](../results/acid_recovery_codex.png).
Full long traces remain in the research workspace; runnable protocols, selected
results and a short full-state reference are distributed here.

## Independent package check

The isolated Windows package reproduced all 4431 states in the short Con2 smoke and Burley 25-pA pulse within 1e-8 absolute difference. Full-state restart also passed. Original research folders, E:/als-tools and network access were blocked during these checks. [Machine-readable result](../provenance/acid_validation_codex.json). Long exported runs were not repeated.

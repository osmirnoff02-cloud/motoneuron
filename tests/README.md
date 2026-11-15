# Verification

The [experiment index](../experiments/README.md) gives the reference simulation and verification commands. Pass `--experiment ID` to require a particular reference. For other configurations, the checker reports its numerical checks and whether it found a matching reference.

[GitHub Actions](../.github/workflows/verify.yml) runs package integrity, phase unit checks, candidate and checkpoint rejection tests, and both smoke experiments on Linux, Windows, and macOS. The longer Burley 25 pA pulse runs on Linux when a release is published or the workflow is started manually. Start the workflow manually to check the pulse before publishing a release.

The local rejection checks use the completed default smoke output.

```bash
python -I -B -X utf8 tests/check_rejected_inputs.py outputs/default-smoke
```

# Releasing FRBench

FRBench versions code and pretrained weights separately:

- Python packages use tags such as `v1.1.0` and are published to PyPI.
- Weight bundles use tags such as `weights-v1.1.0` and are stored as GitHub Release assets.

## Compatibility rules

1. A published weights tag is immutable. Never replace or delete its manifest or archives.
2. Every package version pins the default `release` in `frbench/_config.py` (`_DEFAULTS["release"]`) to the exact weights release tested with that package.
3. Changed weights, model configuration, or manifest entries require a new weights tag. Old package installations therefore continue using their tested assets.
4. Existing manifest keys keep their meaning. If an incompatible model must be introduced, give it a new key or publish a new package version that handles the new format.
5. Release assets and `manifest.json` on GitHub are the source of truth. Do not publish archives from `~/.frbench`, because that cache can contain stale downloads.

An optional top-level manifest entry can communicate future compatibility metadata without being treated as an asset:

```json
{
  "__meta__": {
    "schema_version": 1,
    "min_frbench_version": "1.1.0"
  }
}
```

## Code-only release

Every push to `main` updates the rolling `latest` prerelease and its source archives. To publish a stable package:

1. Update `version` in `pyproject.toml` and `__version__` in `frbench/_config.py` to the same semantic version.
2. Add a `## [x.y.z]` section to `CHANGELOG.md` (the release workflow extracts it as the GitHub release notes) and update the README if needed.
3. Build and test locally:

   ```bash
   python -m pytest tests/ -q
   python -m build
   python -m twine check dist/*
   ```

4. Merge or push the change to `main`.
5. Create and push the matching tag:

   ```bash
   git tag -s v1.1.0 -m "FRBench 1.1.0"
   git push origin v1.1.0
   ```

The `publish.yml` workflow checks version consistency, runs the test suite, builds the wheel and source distribution, publishes them through PyPI Trusted Publishing, and creates the GitHub release `vX.Y.Z` with the built distributions attached as assets (notes taken from the matching `CHANGELOG.md` section).

## Weights release

Prepare every model as a `.tar.gz` archive and produce a new `manifest.json`. Each manifest entry must include the archive name, extracted target directory, expected contents, byte size, and SHA-256 digest. Validate every archive against the manifest before upload.

1. Choose a new, never-before-used tag, for example `weights-v1.1.0`.
2. Create a draft release and upload the manifest and archives:

   ```bash
   gh release create weights-v1.1.0 \
     --repo HKU-TASR/FRBench \
     --draft \
     --title "FR Pretrained Weights v1.1.0" \
     --notes "See the repository README for the model list."
   gh release upload weights-v1.1.0 manifest.json release-assets/*.tar.gz \
     --repo HKU-TASR/FRBench
   ```

3. Download a sample of the uploaded assets into a clean temporary cache and load the corresponding models.
4. Publish the GitHub release. It is immutable from this point onward.
5. Update `_DEFAULTS["release"]` in `frbench/_config.py`, bump the package version, test against the new release, and follow the code release procedure.

Users on older package versions remain pinned to the old weights. Users can test a new release before upgrading by setting `FRBENCH_RELEASE=weights-v1.1.0`, but this override is not guaranteed compatible with an older package.

# conda-forge Submission Guide

This document outlines the steps to submit `pandas-plus` to conda-forge for the first time, and how to maintain it thereafter.

## Initial Submission Process

### 1. Prepare the Package

Before submitting to conda-forge, ensure:
- [ ] Package is published on PyPI
- [ ] Package has a stable version (not dev/pre-release)
- [ ] All dependencies are available on conda-forge
- [ ] Tests pass on multiple platforms
- [ ] Documentation is complete

### 2. Create conda-forge Recipe

Our recipe is already prepared in `conda-recipe/meta.yaml`. Key points:

```yaml
# Required metadata
package:
  name: pandas-plus
  version: "{{ version }}"

# Source from PyPI (conda-forge requirement)
source:
  url: https://pypi.io/packages/source/p/pandas-plus/pandas_plus-{{ version }}.tar.gz
  sha256: {{ sha256 }}

# Dependencies available on conda-forge
requirements:
  run:
    - python >=3.10
    - numpy >=1.19.0
    - pandas >=1.3.0
    - numba >=0.56.0
    - polars >=0.15.0
    - pyarrow >=1.0.0
```

### 3. Submit to conda-forge

#### Option A: Automatic Submission (Recommended)
1. Publish package to PyPI
2. Wait 24-48 hours for conda-forge bot to detect it
3. Bot will create PR at: https://github.com/conda-forge/staged-recipes
4. Monitor for the automated PR and respond to any reviewer feedback

#### Option B: Manual Submission
If automatic detection fails:

1. Fork https://github.com/conda-forge/staged-recipes
2. Create branch: `git checkout -b pandas-plus-recipe`
3. Copy recipe to: `recipes/pandas-plus/meta.yaml`
4. Update sha256 hash from PyPI:
   ```bash
   # Get sha256 from PyPI download
   wget https://pypi.io/packages/source/p/pandas-plus/pandas_plus-0.1.0.tar.gz
   sha256sum pandas_plus-0.1.0.tar.gz
   ```
5. Commit and push:
   ```bash
   git add recipes/pandas-plus/
   git commit -m "Add pandas-plus recipe"
   git push origin pandas-plus-recipe
   ```
6. Create PR to conda-forge/staged-recipes

### 4. Recipe Review Process

Reviewers will check:
- [ ] All dependencies available on conda-forge
- [ ] Recipe follows conda-forge conventions
- [ ] Tests are comprehensive
- [ ] License is correctly specified
- [ ] Package builds on all platforms

Common feedback:
- Simplify test commands
- Add noarch: python if pure Python
- Update dependency versions
- Fix platform-specific issues

## Maintenance Process

After initial acceptance, a feedstock repository will be created at:
`https://github.com/conda-forge/pandas-plus-feedstock`

### Automated Updates
- conda-forge bot monitors PyPI releases
- Automatically creates PRs for version updates
- Updates version and sha256 hash
- Usually no manual intervention needed

### Manual Updates
If automatic updates fail or dependency changes are needed:

1. Fork the feedstock repository
2. Update `recipe/meta.yaml`:
   ```yaml
   {% set version = "0.2.0" %}  # New version
   # Update sha256 hash
   # Update dependencies if needed
   ```
3. Test locally:
   ```bash
   conda build recipe/
   conda install --use-local pandas-plus
   ```
4. Submit PR to feedstock

## GitHub Actions Integration

Our CI/CD workflows support conda-forge:

### Release Workflow (`release.yml`)
- Validates release format
- Builds and tests package
- Publishes to PyPI
- Triggers conda-forge update process

### Conda Build Workflow (`conda-build.yml`)
- Tests conda recipe on all platforms
- Validates conda package functionality
- Ensures conda-forge compatibility

## Checklist for Release

Before each release:
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `conda-recipe/meta.yaml`
- [ ] Update dependencies in both files if changed
- [ ] Run full test suite locally
- [ ] Create GitHub release
- [ ] Monitor PyPI publication
- [ ] Watch for conda-forge bot PR (24-48h)
- [ ] Respond to any conda-forge review feedback

## Troubleshooting

### Common Issues

1. **Bot doesn't create PR**
   - Check if package is on PyPI
   - Verify package name format
   - Look for existing conda-forge package
   - Submit manually to staged-recipes

2. **Build failures**
   - Check platform-specific dependencies
   - Review error logs in PR comments
   - Test locally with conda-build
   - Update recipe constraints

3. **Test failures**
   - Simplify test commands
   - Reduce test dependencies
   - Use import tests instead of full test suite
   - Check for missing test data

### Useful Links

- [conda-forge documentation](https://conda-forge.org/docs/)
- [staged-recipes guidelines](https://github.com/conda-forge/staged-recipes)
- [feedstock maintenance](https://conda-forge.org/docs/maintainer/updating_pkgs.html)
- [conda-build documentation](https://docs.conda.io/projects/conda-build/)

## Contact

- Primary maintainer: @eoincondron
- conda-forge help: https://github.com/conda-forge/conda-forge.github.io/issues
- Gitter chat: https://gitter.im/conda-forge/conda-forge
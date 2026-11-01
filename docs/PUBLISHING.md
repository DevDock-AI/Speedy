# Publishing Speedy

The development branch (`main`) carries internal history (machine-specific commits, personal author identities).
The public repository is pushed from **`publish/v0.1.0`**, a separate history with the same final tree:

| # | Commit | Author |
|---|---|---|
| 1 | Import Strata v0.2.7.1 (Niko1221/Strata, upstream commit 46c152a1) | Strata contributors `<noreply@strata>` |
| 2 | Strata PR #2689/#2690: STRATA_VERIFY_DEFER_CPU (re-ported onto 0.2.7.1) | lask3802 |
| 3 | Strata PR #4283: STRATA_PREFILL_ROUTED_STREAM | Stefano Sinden (original identity) |
| 4-6 | Strata PR #4039: STRATA_PF_GROUP_SCRATCH, STRATA_PF_HC_REGION, docs | merbanan (original identity) |
| 7 | Speedy 0.1.0 (every other difference) | Speedy contributors `<speedy@users.noreply.github.com>` |

`git diff main publish/v0.1.0` is empty when the branch is rebuilt from the current `main`.

## Rebuild after main changes
`tools/make_publish_branch.sh` rebuilds the branch from `main` without touching the work tree (it uses a temporary
index): `tools/make_publish_branch.sh [source-ref] [branch]`, defaults `main publish/v0.1.0`. It refuses to finish
if the result differs from the source tree.

## Set the final author
Pick the identity to publish under (GitHub: Settings > Emails > "Keep my email addresses private" gives
`<id>+<login>@users.noreply.github.com`). Then either rebuild with it:

```sh
SPEEDY_AUTHOR="Your Name <12345+login@users.noreply.github.com>" tools/make_publish_branch.sh
```

or amend only the last commit (the upstream commits keep their authors):

```sh
git checkout publish/v0.1.0
git commit --amend --no-edit --reset-author   # uses your git config user.name / user.email
```

## Push
```sh
git remote add public https://github.com/DevDock-AI/Speedy.git
git push public publish/v0.1.0:main
git tag -a v0.1.0 -m "Speedy 0.1.0" publish/v0.1.0 && git push public v0.1.0
```

Before the first push, replace `OWNER` in `.github/ISSUE_TEMPLATE/config.yml` and add `repository-code` to
`CITATION.cff` (a commit on `main`, then rebuild or cherry-pick it onto the branch). Check that no private
data is in the tree: `git grep -nIiE "srv/projects|@gmail|hostname" publish/v0.1.0`.

## Editions
Each edition is published from its own branch, built the same way (the version names the last commit):

```sh
SPEEDY_VERSION="2GPU 0.1.0" tools/make_publish_branch.sh edition/2gpu publish/2gpu-v0.1.0
SPEEDY_VERSION="1GPU 0.2.0" tools/make_publish_branch.sh edition/1gpu publish/1gpu-v0.2.0
```

Configs carry placeholders instead of machine paths and GPU UUIDs (docs/CONFIG-PLACEHOLDERS.md). Before pushing,
scan the branch: `git grep -nIiE "srv/projects|/srv/models|/root/|@gmail|GPU-[0-9a-f]{8}-" <branch> -- . ':!bench/results'`
(bench/results holds upstream community runs with their own machines' paths).

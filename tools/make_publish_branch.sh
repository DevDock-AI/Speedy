#!/bin/sh
# Build the publishable history (docs/PUBLISHING.md) from a development ref, without touching the work tree:
#   1. Strata v0.2.7.1 import            (Strata contributors)
#   2. PR #2689/#2690 re-port            (lask3802)
#   3. PR #4283                          (original author)
#   4-6. PR #4039 (three commits)        (original author)
#   7. Speedy $SPEEDY_VERSION: everything else ($SPEEDY_AUTHOR)
# The final tree is the source ref's tree; the script fails if it is not.
# Usage: tools/make_publish_branch.sh [source-ref] [branch]   (defaults: main publish/v0.1.0)
set -eu
SRC=${1:-main}
BR=${2:-publish/v0.1.0}
SPEEDY_VERSION=${SPEEDY_VERSION:-0.1.0}
SPEEDY_AUTHOR=${SPEEDY_AUTHOR:-"Speedy contributors <speedy@users.noreply.github.com>"}

BASE=46c152a1      # Niko1221/Strata v0.2.7.1 (merge f-0271-macds)
DEFER=6c93ced2     # PR #2689/#2690 re-port onto the base: its tree is BASE + the PR
PR4283=25100de3
PR4039="340c6c14 ff169f0e 04e1eb45"

idx=$(mktemp)
msg=$(mktemp)
trap 'rm -f "$idx" "$msg"' EXIT
export GIT_INDEX_FILE="$idx"

name_of() { printf '%s' "${1% <*}"; }
mail_of() { printf '%s' "$1" | sed 's/.*<\(.*\)>.*/\1/'; }

# commit <author "Name <mail>"> <date> <parent or ""> : tree from the index, message from $msg, prints the id
commit() {
    tree=$(git write-tree)
    GIT_AUTHOR_NAME=$(name_of "$1") GIT_AUTHOR_EMAIL=$(mail_of "$1") GIT_AUTHOR_DATE="$2" \
    GIT_COMMITTER_NAME=$(name_of "$SPEEDY_AUTHOR") GIT_COMMITTER_EMAIL=$(mail_of "$SPEEDY_AUTHOR") GIT_COMMITTER_DATE="$2" \
        git commit-tree "$tree" ${3:+-p "$3"} -F "$msg"
}

# 1. the upstream import
git read-tree "$BASE"
cat > "$msg" <<EOF
Import Strata v0.2.7.1

The source tree of Strata v0.2.7.1, https://github.com/Niko1221/Strata (MIT), upstream commit
$(git rev-parse "$BASE"). Copyright Niko1221 and the Strata contributors; see LICENSE and NOTICE.
EOF
head=$(commit "Strata contributors <noreply@strata>" "$(git log -1 --format=%aI "$BASE")" "")

# 2. PR #2689/#2690
git read-tree "$DEFER"
cat > "$msg" <<EOF
Strata PR #2689/#2690: STRATA_VERIFY_DEFER_CPU and STRATA_VERIFY_DEFER_CPU_BELOW

Upstream pull requests #2689 and #2690 by lask3802 (Niko1221/Strata, MIT): the verify pass defers the light CPU
expert rows by one layer (opt-in upstream; it changes the output). Re-ported onto v0.2.7.1's cold batches by the
Speedy contributors.
EOF
head=$(commit "lask3802 <lask3802@users.noreply.github.com>" "$(git log -1 --format=%aI "$DEFER")" "$head")

# 3-6. PR #4283 and #4039: each original commit's diff, original author, date and message
for c in $PR4283 $PR4039; do
    git diff "$c^" "$c" | git apply --cached
    git log -1 --format=%B "$c" > "$msg"
    case " $PR4039 " in *" $c "*) pr=4039 ;; *) pr=4283 ;; esac
    printf '\nUpstream: Niko1221/Strata PR #%s\n' "$pr" >> "$msg"
    head=$(commit "$(git log -1 --format='%an <%ae>' "$c")" "$(git log -1 --format=%aI "$c")" "$head")
done

# 7. everything else
git read-tree "$SRC"
cat > "$msg" <<EOF
Speedy $SPEEDY_VERSION

A measured, tuned fork of Strata v0.2.7.1 for Qwen3.8-Flash-Next on consumer Ampere GPUs: built-in production
defaults (docs/SPEEDY_DEFAULTS.md), presets, the benchmark and quality-gate harness, J13 pin-by-need pricing,
opt-in PSI shedding, the Code Mode MCP tool, docs and CI. See CHANGELOG.md and SPEEDY.md.
EOF
head=$(commit "$SPEEDY_AUTHOR" "$(git log -1 --format=%cI "$SRC")" "$head")

if [ -n "$(git diff-tree -r "$SRC" "$head")" ]; then
    echo "make_publish_branch: the result differs from $SRC" >&2
    exit 1
fi
git update-ref "refs/heads/$BR" "$head"
echo "$BR -> $head (tree = $SRC's tree)"
git log --format='%h %an <%ae> %s' "$BR"

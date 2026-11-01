#!/bin/bash
# dual_ab.sh <tag> arm... : Phase 2 dual-GPU A/B (ab4 harness). Arms = tune/<arm>.json, e.g.
#   dual_ab.sh s1 dual-q2-ctl dual-q2-split dual-q2-split dual-q2-ctl      (ABBA vs single GPU1, same build-dual exe)
# Per arm: cold 60K, 120K (+warm 2K), 200K; then per-card VRAM (GPU1 first) + split/helper/peer planner lines. Under GPU.lock.
TAG=$1; shift; S=<CONFIG_DIR>; L=<SCRIPTS_DIR>; B=<BENCH_DIR>; T=$S/tune
EXE=<SPEEDY_DIR>/build/strata
[ -x $EXE ] || { echo "no $EXE"; exit 1; }
exec 9>$B/GPU.lock; flock 9
trap 'systemctl stop strata-llm; $L/clean.sh >/dev/null; systemctl start llama-swap; echo "llama-swap restored"; flock -u 9' EXIT
$B/phase/idle.sh || exit 1; systemctl stop llama-swap
echo "engine sha256 $(sha256sum $EXE | cut -c1-16) ($(git -C <SPEEDY_DIR> rev-parse --short HEAD))"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,power.limit,clocks.max.sm,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
vram() { nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader | sed 's/${GPU1_UUID:-<GPU1_UUID>}/GPU1/;s/${GPU2_UUID:-<GPU2_UUID>}/GPU2/' | tr '\n' ' '; }
i=0; for a in "$@"; do i=$((i+1)); echo "== $a #$i $(date -Is)"; $L/start_cfg.sh $T/$a.json || { echo "START FAIL $a"; tail -8 $T/$a.log; continue; }
  grep -E "layer split|split search|split at|stage|trim|draft head|peer|helper|CUDA1|prompt chunk|PREFILL_MAX|pipeline-windows|P2P" $T/$a.log | head -14 | cut -c1-200
  echo "vram loaded: $(vram)"
  python3 $B/speedbench.py $a-$TAG$i-k60 $T/$a.log --k 55 --reps 1 | tail -1
  python3 $B/speedbench.py $a-$TAG$i-k120 $T/$a.log --k 111 --reps 1 --warm 2000 | grep -E "speed"
  timeout 1200 python3 $B/speedbench.py $a-$TAG$i-k200 $T/$a.log --k 185 --reps 1 | tail -1
  echo "vram after 200K: $(vram)"; grep -c -i "wait" $T/$a.log | sed 's/^/wait lines: /'
done
echo "Xid $(journalctl -k -b --no-pager | grep -c Xid)"

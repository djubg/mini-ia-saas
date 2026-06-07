# train_journee.ps1 — À LANCER LE MATIN AVANT DE PARTIR.
#
# Entraîne ~8h sur le GPU (DirectML), PUIS évalue et génère automatiquement.
# Quand tu rentres, tout est déjà fait et logué dans checkpoints/journee_*.log.
#
# Lancer :  .\train_journee.ps1
# (la data doit déjà être préparée : data/train.bin, val.bin, tokenizer.json)

$ErrorActionPreference = "Continue"
$log = "checkpoints/journee_$(Get-Date -Format yyyyMMdd_HHmm).log"
Write-Host "Log complet -> $log" -ForegroundColor Cyan

# Empêche la mise en veille pendant la journée (sinon le GPU s'arrête).
try { powercfg /change standby-timeout-ac 0 } catch { Write-Host "(pense à désactiver la veille manuellement)" -ForegroundColor Yellow }

Write-Host "`n========== 1/3  ENTRAÎNEMENT (40000 iters) ==========" -ForegroundColor Green
python src/train.py --device dml --n_layer 8 --n_head 8 --n_embd 512 --block_size 256 `
    --batch_size 8 --grad_accum 2 --max_iters 40000 --eval_interval 1000 2>&1 | Tee-Object -FilePath $log

Write-Host "`n========== 2/3  ÉVALUATION ==========" -ForegroundColor Green
python src/eval.py --device dml 2>&1 | Tee-Object -FilePath $log -Append

Write-Host "`n========== 3/3  GÉNÉRATION ==========" -ForegroundColor Green
python src/generate.py --prompt "How to build a SaaS product" --device cpu `
    --max_new_tokens 150 --top_p 0.9 --repetition_penalty 1.3 --stop 2>&1 | Tee-Object -FilePath $log -Append

Write-Host "`nTerminé. Tout est dans $log" -ForegroundColor Cyan
Write-Host "Modèle : checkpoints/ckpt.pt (meilleur) et ckpt_last.pt (dernier)."

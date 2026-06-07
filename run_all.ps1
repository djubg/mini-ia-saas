# run_all.ps1 — Pipeline complet en une commande.
#
# Enchaîne : (filtrage) -> tokenisation -> entraînement -> évaluation.
# La COLLECTE reste séparée (elle dépend de la source/taille voulue) :
#   python src/collect_big_data.py --target_gb 2 --source fineweb --filter_saas
#
# Exemples :
#   .\run_all.ps1                          # tokenise data/ + entraîne (défauts) + évalue
#   .\run_all.ps1 -Filter -Retrain         # filtre data/big, réentraîne le tokenizer, entraîne
#   .\run_all.ps1 -MaxIters 15000 -Device dml
#
param(
    [int]$MaxIters = 8000,
    [string]$Device = "dml",      # dml (GPU AMD) | cpu | auto
    [switch]$Filter,              # filtrer data/big -> data/big_saas avant de tokeniser
    [switch]$Retrain              # réentraîner le tokenizer BPE
)
$ErrorActionPreference = "Stop"
$py = "python"

if ($Filter) {
    if (Test-Path "data/big") {
        Write-Host "`n== 1/4  Filtrage SaaS strict (data/big -> data/big_saas) ==" -ForegroundColor Cyan
        & $py src/filter_corpus.py --in_dir data/big --out_dir data/big_saas
    } else {
        Write-Host "data/big absent — filtrage ignoré." -ForegroundColor Yellow
    }
}

Write-Host "`n== 2/4  Tokenisation ==" -ForegroundColor Cyan
$prep = @("src/prepare_data.py", "--no_big")
if ($Retrain) { $prep += "--retrain_tokenizer" }
& $py @prep

Write-Host "`n== 3/4  Entraînement ($Device, $MaxIters iters) ==" -ForegroundColor Cyan
& $py src/train.py --device $Device --max_iters $MaxIters

Write-Host "`n== 4/4  Évaluation ==" -ForegroundColor Cyan
& $py src/eval.py --device $Device

Write-Host "`nTerminé. Génère du texte avec :" -ForegroundColor Green
Write-Host '  python src/generate.py --prompt "How to build a SaaS product" --top_p 0.9 --repetition_penalty 1.3 --stop'

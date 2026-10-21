# Physics v1 (ADR-0011) re-train + evaluation on the PC (label win-isaac). Sequential; a failed run is logged and the
# pipeline continues (scenario fine-tunes are skipped only if v1_nominal produced no checkpoint).
# usage: powershell -File runs\ppo_win-isaac\eval_v1\run_v1.ps1 [-Stage train|eval|all]
param([string]$Stage = "all")
$ErrorActionPreference = "Continue"
Set-Location D:\Workspace\ai-rocket-ppo
$Lab = "C:\dev\IsaacLab\isaaclab.bat"
$T = "runs\ppo_win-isaac\train"
$E = "runs\ppo_win-isaac\eval_v1"
$Rew = @("--shaping", "guided", "--phi_pos_m", "2.0", "--phi_tilt_rad", "0.3", "--rew_td_success", "20")  # ADR-0010
$Status = "$E\status.jsonl"

function Note($o) { ($o | ConvertTo-Json -Compress) | Add-Content -Encoding utf8 $Status }

function Latest($run) {
    $f = Get-ChildItem "$T\$run\model_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.BaseName -replace "model_", "") } | Select-Object -Last 1
    if ($f) { return $f.FullName } else { return $null }
}

function Train($run, [string[]]$extra) {
    $t0 = Get-Date
    & $Lab -p scripts\ppo_train.py --run_name $run --num_envs 4096 --physics v1 @Rew @extra --headless > "$T\..\$run.log" 2>&1
    $ck = Latest $run
    $done = [bool](Select-String -Path "$T\..\$run.log" -Pattern "TRAIN_DONE" -Quiet)
    Note @{ step = "train"; run = $run; args = ($extra -join " "); exit = $LASTEXITCODE; train_done = $done;
            checkpoint = $ck; wall_s = [int]((Get-Date) - $t0).TotalSeconds }
    return $ck
}

function Eval($scen, $name, [string[]]$extra) {
    foreach ($s0 in 1000, 2000) {
        $out = "$E\$scen\$name\s$s0"
        New-Item -ItemType Directory -Force $out | Out-Null
        & $Lab -p scripts\ppo_eval.py --physics v1 --scenario $scen --n 30 --seed0 $s0 --out $out @extra --headless > "$out\eval.log" 2>&1
        Note @{ step = "eval"; scenario = $scen; controller = $name; seed0 = $s0; exit = $LASTEXITCODE;
                ok = [bool](Select-String -Path "$out\eval.log" -Pattern "EVAL_RESULT" -Quiet) }
    }
}

$Scen = @{ wind = 300; rcs_fail = 300; noise = 300; lateral = 400 }
$Layouts = "tri_diff", "quad_diff", "coldgas"

if ($Stage -in "train", "all") {
    $nom = Train "v1_nominal" @("--max_iterations", "300", "--resume_from", "$T\r3_guided\model_398.pt")
    foreach ($s in "wind", "rcs_fail", "noise", "lateral") {
        if ($nom) { Train "v1_$s" @("--scenario", $s, "--max_iterations", "$($Scen[$s])", "--resume_from", $nom) | Out-Null }
        else { Note @{ step = "train"; run = "v1_$s"; skipped = "no v1_nominal checkpoint" } }
    }
    foreach ($l in $Layouts) { Train "v1_$l" @("--layout", $l, "--max_iterations", "600") | Out-Null }
}

if ($Stage -in "eval", "all") {
    $nom = Latest "v1_nominal"
    foreach ($s in "nominal", "wind", "rcs_fail", "noise", "lateral") {
        Eval $s "pd" @("--controller", "pd")
        $g = if ($s -eq "wind") { "runs\pd_tuning\wind_try1_8seeds.json" } else { "runs\pd_tuning\$s.json" }
        Eval $s "pd_tuned" @("--controller", "pd", "--pd_gains", $g)
        if ($nom) { Eval $s "rl_nominal" @("--controller", "ppo", "--checkpoint", $nom) }
        if ($s -ne "nominal") {
            $ck = Latest "v1_$s"
            if ($ck) { Eval $s "rl_scenario" @("--controller", "ppo", "--checkpoint", $ck) }
        }
    }
    foreach ($l in $Layouts) {
        $ck = Latest "v1_$l"
        if ($ck) { Eval "nominal" "rl_$l" @("--controller", "ppo", "--checkpoint", $ck, "--layout", $l) }
    }
}
Note @{ step = "pipeline_done"; stage = $Stage }

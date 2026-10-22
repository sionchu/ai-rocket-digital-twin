# Ground-start hop (ADR-0012), physics v1, label win-isaac: PPO fine-tune from v1_nominal, then 60-seed evaluation
# of the default PD (hover mode, then descent) and the hop policy.
$ErrorActionPreference = "Continue"
Set-Location D:\Workspace\ai-rocket-ppo
$Lab = "C:\dev\IsaacLab\isaaclab.bat"
$T = "runs\ppo_win-isaac\train"
$E = "runs\ppo_win-isaac\eval_v1"
& $Lab -p scripts\ppo_train.py --run_name v1_hop --scenario hop --max_iterations 600 --num_envs 4096 --physics v1 `
    --shaping guided --phi_pos_m 2.0 --phi_tilt_rad 0.3 --rew_td_success 20 `
    --resume_from "$T\v1_nominal\model_697.pt" --headless > runs\ppo_win-isaac\v1_hop.log 2>&1
$ck = Get-ChildItem "$T\v1_hop\model_*.pt" | Sort-Object { [int]($_.BaseName -replace "model_", "") } | Select-Object -Last 1
"train done: $($ck.Name)"
foreach ($c in @(@("pd", @("--controller", "pd")), @("rl_scenario", @("--controller", "ppo", "--checkpoint", $ck.FullName)))) {
    foreach ($s0 in 1000, 2000) {
        $out = "$E\hop\$($c[0])\s$s0"
        New-Item -ItemType Directory -Force $out | Out-Null
        & $Lab -p scripts\ppo_eval.py --physics v1 --scenario hop --n 30 --seed0 $s0 --out $out @($c[1]) --headless > "$out\eval.log" 2>&1
        "$($c[0]) $s0 $(Select-String -Path "$out\eval.log" -Pattern 'EVAL_RESULT' -Quiet)"
    }
}
"HOP_DONE"

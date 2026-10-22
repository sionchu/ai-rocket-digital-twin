# Isaac footage for the physics-v1 story cut (render_story.py --cut v3): physics v1 + visual v1, label win-isaac.
# Seeds are named on screen; outcome rates per seed block are in runs/ppo_win-isaac/eval_v1.
$ErrorActionPreference = "Continue"
Set-Location D:\Workspace\ai-rocket-ppo
$Lab = "C:\dev\IsaacLab\isaaclab.bat"
$T = "runs\ppo_win-isaac\train"
$V = @("--physics", "v1", "--visual", "v1", "--look", "story", "--enable_cameras", "--headless")

function M($name, [string[]]$a) {
    & $Lab -p scripts\isaac_mission_video.py --name $name @a @V > "runs\ppo_win-isaac\mission_$name.log" 2>&1
    "$name $(Select-String -Path "runs\ppo_win-isaac\mission_$name.log" -Pattern 'MISSION_DONE' -Quiet)"
}
function G($name, [string[]]$a) {
    & $Lab -p scripts\isaac_grid_video.py --name $name @a @V > "runs\ppo_win-isaac\grid_$name.log" 2>&1
    "$name $(Select-String -Path "runs\ppo_win-isaac\grid_$name.log" -Pattern 'GRID_DONE|DONE' -Quiet)"
}

G "g_v1_r3" @("--checkpoint", "$T\r3_guided\model_398.pt")
G "g_v1_nom" @("--checkpoint", "$T\v1_nominal\model_697.pt")
M "v1_wind_pd" @("--controller", "pd", "--scenario", "wind", "--seed", "1001")
M "v1_wind_rl" @("--controller", "ppo", "--checkpoint", "$T\v1_wind\model_996.pt", "--scenario", "wind", "--seed", "1001")
M "v1_lat_pdt" @("--controller", "pd", "--pd_gains", "runs\pd_tuning\lateral.json", "--scenario", "lateral", "--seed", "1002")
M "v1_lat_rl" @("--controller", "ppo", "--checkpoint", "$T\v1_lateral\model_1096.pt", "--scenario", "lateral", "--seed", "1002")
M "v1_rcs_rl" @("--controller", "ppo", "--checkpoint", "$T\v1_rcs_fail\model_996.pt", "--scenario", "rcs_fail", "--seed", "1000")
M "v1_noise_rl" @("--controller", "ppo", "--checkpoint", "$T\v1_noise\model_996.pt", "--scenario", "noise", "--seed", "1000")
M "v1_ly_a" @("--controller", "ppo", "--checkpoint", "$T\v1_nominal\model_697.pt", "--seed", "1000")
M "v1_ly_b" @("--controller", "ppo", "--checkpoint", "$T\v1_tri_diff\model_599.pt", "--layout", "tri_diff", "--seed", "1000")
M "v1_ly_c" @("--controller", "ppo", "--checkpoint", "$T\v1_quad_diff\model_599.pt", "--layout", "quad_diff", "--seed", "1000")
M "v1_ly_d" @("--controller", "ppo", "--checkpoint", "$T\v1_coldgas\model_599.pt", "--layout", "coldgas", "--seed", "1000")
# cut v4 cold open: Flow flame (brighter for the daylight sky) + engine camera
M "v1_hero_wind" @("--controller", "ppo", "--checkpoint", "$T\v1_wind\model_996.pt", "--scenario", "wind", "--seed", "1001",
    "--plume", "flow", "--flow_set", "flowRender/rayMarch:colorScale=20", "flowOffscreen/colormap:colorScale=20",
    "box:temperature=3", "box:fuel=3", "--engine_cam", "--cam_dist", "18")
# cut v5 cold open: ground-start hop (ADR-0012)
M "v1_hero_hop" @("--controller", "ppo", "--checkpoint", "$T\v1_hop\model_1296.pt", "--scenario", "hop", "--seed", "1000",
    "--plume", "flow", "--flow_set", "flowRender/rayMarch:colorScale=20", "flowOffscreen/colormap:colorScale=20",
    "box:temperature=3", "box:fuel=3", "--engine_cam", "--cam_dist", "22")
"RENDER_V1_DONE"

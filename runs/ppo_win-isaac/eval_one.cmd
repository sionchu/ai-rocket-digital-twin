@echo off
cd /d D:\Workspace\ai-rocket-ppo
C:\dev\IsaacLab\isaaclab.bat -p scripts\ppo_eval.py --checkpoint runs\ppo_win-isaac\train\%1.pt --out runs\ppo_win-isaac\train\%1_eval %2 %3 %4 %5 %6 > runs\ppo_win-isaac\train\%1_eval.log 2>&1

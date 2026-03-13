#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 环境诊断工具：检查 DISPLAY、Tkinter/_tkinter、Matplotlib 后端，并给出修复建议。
"""
import os
import sys
import platform
import traceback


def check_env():
    print("== 环境变量 ==")
    print(f"DISPLAY={os.environ.get('DISPLAY', '')}")
    print(f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')}")
    print(f"XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE', '')}")
    print()


def check_python():
    print("== Python 信息 ==")
    print(f"python={sys.version}")
    print(f"platform={platform.platform()}")
    print()


def check_tk():
    print("== Tkinter 检查 ==")
    try:
        import tkinter as tk
        print("tkinter: 已导入")
    except Exception as e:
        print(f"tkinter: 导入失败 -> {e}")
        traceback.print_exc()
        print("建议：安装 Tk 支持：\n- apt 系统：sudo apt-get update && sudo apt-get install -y python3-tk\n- conda 环境：conda install -c conda-forge tk")
        print()
        return

    try:
        import _tkinter  # noqa: F401
        print("_tkinter: 已导入")
    except Exception as e:
        print(f"_tkinter: 导入失败 -> {e}")
        traceback.print_exc()
        print("建议：安装 Tk 支持：\n- apt 系统：sudo apt-get update && sudo apt-get install -y python3-tk\n- conda 环境：conda install -c conda-forge tk")
        print()
        return

    # 尝试创建 Tk 根窗口
    try:
        r = tk.Tk()
        print("Tk: 成功创建根窗口")
        r.withdraw()
        r.destroy()
    except Exception as e:
        print(f"Tk: 创建失败 -> {e}")
        traceback.print_exc()
        print("可能原因：\n- 无图形环境（DISPLAY 未设置）\n- 远程/容器环境未配置 X11 转发或虚拟显示")
        print("可选解决方案：\n- 在桌面环境直接运行：python3 main.py\n- SSH 图形转发：ssh -X <host>，然后运行程序\n- 使用虚拟显示：xvfb-run -a python3 main.py\n- 在容器中映射 X11：挂载 /tmp/.X11-unix 并设置 DISPLAY")
    print()


def check_matplotlib():
    print("== Matplotlib 后端 ==")
    try:
        import matplotlib as mpl
        print(f"matplotlib={mpl.__version__}")
        print(f"backend={mpl.get_backend()}")
        print("提示：GUI 使用 TkAgg 后端；若无图形环境可改用 Agg 进行离线渲染。")
    except Exception as e:
        print(f"matplotlib: 导入失败 -> {e}")
        traceback.print_exc()
    print()


def advice_summary():
    disp = os.environ.get('DISPLAY')
    xdg = os.environ.get('XDG_SESSION_TYPE')
    print("== 修复建议汇总 ==")
    if not disp:
        print("- 检测到 DISPLAY 为空，当前可能是无图形环境或终端会话。")
        print("  方案 A：使用桌面环境直接运行 GUI。")
        print("  方案 B：SSH 图形转发：ssh -X <host>；确认 echo $DISPLAY 非空。")
        print("  方案 C：在服务器/容器使用虚拟显示：xvfb-run -a python3 main.py。")
    if (xdg or '').lower() in ('tty', 'unknown', ''):
        print(f"- XDG_SESSION_TYPE={xdg}，通常表示非图形会话。建议采用上述 B/C 方案。")
    print("- 若报 _tkinter 或 tkinter 导入失败，安装 Tk 支持：apt 安装 python3-tk 或 conda 安装 tk。")
    print()


def main():
    check_python()
    check_env()
    check_tk()
    check_matplotlib()
    advice_summary()


if __name__ == '__main__':
    main()
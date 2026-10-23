#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import atexit
import os
import re
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, redirect, request

ROOT = Path(__file__).resolve().parent
LAUNCHER_PORT = int(os.environ.get("FYP_LAUNCHER_PORT", "8787"))
TOKEN = os.environ.get("FYP_LAUNCHER_TOKEN", "").strip()
FACE_APP_URL = os.environ.get("FYP_FACEAPP_CHECK", "http://127.0.0.1:5000/api/status")

def _face_app_ui_url():
    override = (os.environ.get("FYP_FACE_UI") or "").strip().rstrip("/")
    if override:
        return override + "/"
    check = FACE_APP_URL.rstrip("/")
    suffix = "/api/status"
    if check.lower().endswith(suffix.lower()):
        return check[: -len(suffix)] + "/"
    return "http://127.0.0.1:5000/"

_BUILTIN_DEFAULT_RADAR = "http://10.200.179.43:5000"
DEFAULT_RADAR_PI_BASE = (
    os.environ.get("FYP_DEFAULT_RADAR_PI_BASE") or _BUILTIN_DEFAULT_RADAR
).strip()

launcher_app = Flask(__name__)

_spawned_face_proc = None
_face_app_log_fh = None

_FACE_APP_BOOT_LOG = ROOT / ".face_app_boot.log"

def _face_app_port():
    parsed = urllib.parse.urlparse(FACE_APP_URL)
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80

def _pids_listening_on_port(port: int):
    pids = []
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception:
            return []
        needle = ":%d" % port
        for line in (out.stdout or "").splitlines():
            if "LISTENING" not in line.upper() or needle not in line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                pids.append(int(parts[-1]))
            except ValueError:
                continue
    else:
        try:
            out = subprocess.run(
                ["ss", "-ltnp"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception:
            try:
                out = subprocess.run(
                    ["lsof", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                for token in (out.stdout or "").split():
                    try:
                        pids.append(int(token.strip()))
                    except ValueError:
                        pass
                return sorted(set(pids))
            except Exception:
                return []
        for line in (out.stdout or "").splitlines():
            m = re.search(r"pid=(\d+)", line)
            if m and (":%d" % port) in line:
                pids.append(int(m.group(1)))
    return sorted(set(pids))

def _pid_commandline(pid: int) -> str:
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine"
                    % pid,
                ],
                capture_output=True,
                text=True,
                timeout=12,
                check=False,
            )
            return (r.stdout or "").strip()
        except Exception:
            return ""
    try:
        raw = Path("/proc/%d/cmdline" % pid).read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return ""

def _is_face_app_pid(pid: int) -> bool:
    cmd = _pid_commandline(pid).replace("\\", "/").lower()
    return "face_app.py" in cmd

def _kill_pid_tree(pid: int):
    if pid <= 0:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1.0)
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    except Exception:
        pass

def _cleanup_stale_face_listeners():
    port = _face_app_port()
    killed = []
    for pid in _pids_listening_on_port(port):
        if not _is_face_app_pid(pid):
            continue
        print("[INFO] 发现残留 face_app（PID %s 占用端口 %s），正在结束…" % (pid, port))
        _kill_pid_tree(pid)
        killed.append(pid)
    if killed:
        time.sleep(1.0)
    return killed

def _face_app_running():
    try:
        urllib.request.urlopen(FACE_APP_URL, timeout=1.5)
        return True
    except Exception:
        return False

def _merge_env_for_face_app():
    env = os.environ.copy()
    if not (env.get("RADAR_PI_BASE") or "").strip():
        env["RADAR_PI_BASE"] = DEFAULT_RADAR_PI_BASE
        print("[INFO] 未检测到 RADAR_PI_BASE，已为 face_app 注入默认: %s" % env["RADAR_PI_BASE"])
        print("       （可在本终端先执行 $env:RADAR_PI_BASE=\"...\" 覆盖）")
    return env

def _open_face_app_log():
    global _face_app_log_fh
    if _face_app_log_fh is not None:
        return _face_app_log_fh
    _face_app_log_fh = open(_FACE_APP_BOOT_LOG, "a", encoding="utf-8", buffering=1)
    _face_app_log_fh.write(
        "\n--- face_app boot %s ---\n" % time.strftime("%Y-%m-%d %H:%M:%S")
    )
    _face_app_log_fh.flush()
    return _face_app_log_fh

def _close_face_app_log():
    global _face_app_log_fh
    if _face_app_log_fh is not None:
        try:
            _face_app_log_fh.close()
        except Exception:
            pass
        _face_app_log_fh = None

def _popen_face_app():
    script = ROOT / "face_app.py"
    if not script.is_file():
        raise FileNotFoundError("找不到 face_app.py: %s" % script)
    env = _merge_env_for_face_app()
    kwargs = {"cwd": str(ROOT), "env": env}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    log = _open_face_app_log()
    kwargs["stdout"] = log
    kwargs["stderr"] = subprocess.STDOUT
    proc = subprocess.Popen([sys.executable, str(script)], **kwargs)
    global _spawned_face_proc
    _spawned_face_proc = proc
    return proc

def _terminate_spawned_face_app():
    global _spawned_face_proc
    if os.environ.get("FYP_EXIT_KILL_FACE", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        print("[INFO] 已设置 FYP_EXIT_KILL_FACE=0，保留 face_app 进程")
        _spawned_face_proc = None
        _close_face_app_log()
        return
    proc = _spawned_face_proc
    _spawned_face_proc = None
    try:
        if proc is not None:
            pid = proc.pid
            print("[INFO] 正在结束本程序启动的 face_app（PID %s）..." % pid)
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                _kill_pid_tree(pid)
            except Exception as e:
                print("[WARN] 结束 face_app 时出现问题: %s" % e)
        swept = _cleanup_stale_face_listeners()
        if swept and proc is None:
            print(
                "[INFO] 已结束本机 face_app（PID %s）"
                % ", ".join(str(p) for p in swept)
            )
    finally:
        _close_face_app_log()

def _wait_face_app(timeout_sec=120, step_sec=1.0):
    deadline = time.time() + timeout_sec
    last_report = 0.0
    while time.time() < deadline:
        if _face_app_running():
            return True
        now = time.time()
        if now - last_report >= 12.0:
            waited = int(now - (deadline - timeout_sec))
            print(
                "[INFO] 仍在等待 face_app 监听 %s …（已等待约 %ss，首次启动会在后台加载深度学习模型）"
                % (FACE_APP_URL, waited)
            )
            last_report = now
        time.sleep(step_sec)
    return False

def ensure_face_app_at_boot():
    if os.environ.get("FYP_SKIP_FACE_APP_BOOT", "").strip() in ("1", "true", "yes"):
        print("[INFO] 已设置 FYP_SKIP_FACE_APP_BOOT，跳过自动启动 face_app")
        return
    if _face_app_running():
        print("[OK] face_app 已在运行  Web: %s" % _face_app_ui_url().rstrip("/"))
        return
    port = _face_app_port()
    busy = _pids_listening_on_port(port)
    if busy:
        foreign = [p for p in busy if not _is_face_app_pid(p)]
        if foreign:
            print(
                "[ERROR] 端口 %s 已被其他程序占用（PID %s），无法启动 face_app。"
                % (port, ", ".join(str(p) for p in foreign))
            )
            return
        _cleanup_stale_face_listeners()
    print("[INFO] 正在启动 face_app.py（与本 launcher 使用同一 Python: %s）" % sys.executable)
    print("[INFO] face_app 运行日志: %s" % _FACE_APP_BOOT_LOG)
    try:
        _popen_face_app()
    except Exception as e:
        print("[ERROR] 无法启动 face_app: %s" % e)
        return
    if _wait_face_app():
        print("[OK] face_app 已就绪  浏览器: %s" % _face_app_ui_url().rstrip("/"))
    else:
        proc = _spawned_face_proc
        exit_code = proc.poll() if proc is not None else None
        hint = ""
        if exit_code is not None:
            hint = " face_app 已退出（code=%s）。" % exit_code
        print(
            "[WARN] %s 秒内未检测到 face_app。%s"
            % (120, hint)
            + " 请查看 %s 或手动运行: python face_app.py"
            % _FACE_APP_BOOT_LOG
        )

@launcher_app.route("/")
def launcher_root():
    return redirect(_face_app_ui_url(), code=302)

@launcher_app.route("/health")
def health():
    return jsonify({"ok": True, "face_app": _face_app_running()})

@launcher_app.route("/launch", methods=["POST"])
def launch():
    if TOKEN:
        if request.headers.get("X-FYP-Token") != TOKEN:
            return jsonify({"ok": False, "error": "未授权"}), 403
    if _face_app_running():
        return jsonify({"ok": True, "message": "face_app 已在运行", "started": False})

    _cleanup_stale_face_listeners()

    script = ROOT / "face_app.py"
    if not script.is_file():
        return jsonify({"ok": False, "error": "找不到 face_app.py"}), 500

    try:
        _popen_face_app()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    if _wait_face_app(timeout_sec=90):
        return jsonify({"ok": True, "message": "face_app 已启动", "started": True})
    return jsonify(
        {
            "ok": False,
            "error": "已尝试启动但 90 秒内未检测到 face_app，请在本机手动运行 python face_app.py 查看报错",
            "started": True,
        }
    )

if __name__ == "__main__":
    print("=" * 60)
    print("FYP 上位机一键启动")
    print("  Python: %s" % sys.executable)
    print("  工程目录: %s" % ROOT)
    print("=" * 60)
    ensure_face_app_at_boot()
    print("-" * 60)
    print("launcher 监听 0.0.0.0:%s   GET / → 跳转 Web 界面；POST /launch（树莓派可唤醒）" % LAUNCHER_PORT)
    print("Web 界面（face_app）: %s   就绪检测: %s" % (_face_app_ui_url().rstrip("/"), FACE_APP_URL))
    if TOKEN:
        print("已启用令牌: 树莓派设置相同 FYP_LAUNCHER_TOKEN")
    print("按 Ctrl+C 退出 launcher（将同时结束由本程序启动的 face_app）")
    print("=" * 60)
    atexit.register(_terminate_spawned_face_app)

    try:
        launcher_app.run(host="0.0.0.0", port=LAUNCHER_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print()
    finally:
        _terminate_spawned_face_app()

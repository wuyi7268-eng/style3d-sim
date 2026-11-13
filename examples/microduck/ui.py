"""Chinese local web workbench: python -m examples.microduck.ui [--login]."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .ui_backend import ROOT, Workbench, catalog

ASSETS = Path(__file__).with_name("ui_assets")


def configuration():
    from .env import ENVIRONMENT_CONFIGS
    from .ppo import apply_official_ppo, ppo_config
    from .task_profiles import load_task_spec
    from .task_registry import TASKS

    return {
        "tasks": list(TASKS),
        "profiles": {name: asdict(value) for name, value in ENVIRONMENT_CONFIGS.items()},
        "ppo": {task: apply_official_ppo(ppo_config(1024), load_task_spec(task)[0], 1024) for task in TASKS},
    }


def make_server(workbench, port, metadata):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            # Access logs contain no useful training information and can expose URL data.
            pass

        def respond(self, status, body, mime="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Navigation can cancel an in-flight status request after the response was built.
                pass

        def authorized(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in allowed:
                self.respond(403, {"error": "仅接受本机地址"})
                return False
            if self.path.startswith("/api/") and not secrets.compare_digest(self.headers.get("X-Workbench-Token", ""), token):
                self.respond(403, {"error": "请刷新控制台页面后重试"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            path = urlsplit(self.path).path
            if path == "/api/config":
                self.respond(
                    200,
                    {
                        **metadata,
                        "api_version": 2,
                        "log_root": str(workbench.log_root),
                        "synreal_ready": all(workbench.child_env.get(key) for key in ("STYLE3D_ACCOUNT", "STYLE3D_PASSWORD")),
                    },
                )
            elif path == "/api/status":
                log_id = parse_qs(urlsplit(self.path).query).get("log_id", [""])[0]
                self.respond(200, workbench.snapshot(log_id=log_id))
            elif path == "/api/models":
                self.respond(200, catalog(workbench.log_root))
            elif path in ("/", "/app.js", "/style.css"):
                name = "index.html" if path == "/" else path[1:]
                mime = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
                body = (ASSETS / name).read_bytes().replace(b"__WORKBENCH_TOKEN__", token.encode())
                self.respond(200, body, mime + "; charset=utf-8")
            else:
                self.respond(404, {"error": "没有这个页面"})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("请求长度无效")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise TypeError("请求必须为 JSON 对象")
                actions = {
                    "/api/train": lambda: workbench.train(data),
                    "/api/preview": lambda: workbench.play(data, preview=True),
                    "/api/evaluate": lambda: workbench.play(data),
                    "/api/stop": lambda: workbench.stop(data.get("job_id")),
                    "/api/tensorboard": workbench.tensorboard,
                }
                if self.path not in actions:
                    self.respond(404, {"error": "未知操作"})
                    return
                result = actions[self.path]()
                self.respond(200, result or {"ok": True})
            except (ValueError, TypeError, OSError, KeyError) as exc:
                self.respond(400, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="MicroDuck 中文训练与评估控制台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-root", type=Path, default=ROOT / "logs/rsl_rl")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--login", action="store_true", help="在终端隐藏输入 SynReal 凭据，仅保留于进程内存")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    child_env = dict(os.environ)
    if args.login:
        child_env["STYLE3D_ACCOUNT"] = getpass.getpass("Style3D account (hidden): ").strip()
        child_env["STYLE3D_PASSWORD"] = getpass.getpass("Style3D password (hidden): ")
        if not child_env["STYLE3D_ACCOUNT"] or not child_env["STYLE3D_PASSWORD"]:
            parser.error("account and password are required")
    workbench = Workbench(args.log_root, child_env)
    server = make_server(workbench, args.port, configuration())
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"MicroDuck 控制台: {url}\n关闭浏览器不停止训练；终端 Ctrl+C 会停止本控制台启动的进程。", flush=True)
    if not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n正在关闭训练和可视化进程，保留已经保存的权重…", flush=True)
    finally:
        server.server_close()
        workbench.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

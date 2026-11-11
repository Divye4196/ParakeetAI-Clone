import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_engine import LLMEngine, classify_question  # noqa: E402


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        if req.get("model") == "missing":
            body = b'{"error":"model not found"}'
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        words = ("ANSWER:" + req["prompt"]).split() + ["end"]
        delay = 0.25 if "slow" in req["prompt"] else 0.01
        try:
            for w in words:
                line = (json.dumps({"response": w + " ", "done": False}) + "\n").encode()
                self.wfile.write(("%x\r\n" % len(line)).encode() + line + b"\r\n")
                self.wfile.flush()
                time.sleep(delay)
            line = (json.dumps({"response": "", "done": True}) + "\n").encode()
            self.wfile.write(("%x\r\n" % len(line)).encode() + line + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/api/generate" % srv.server_address[1]


def collect(engine, question, wait=10):
    tokens, result, done = [], [], threading.Event()
    engine.start_answer(
        question,
        lambda i, t: tokens.append((i, t)),
        lambda i: (result.append(("done", i)), done.set()),
        lambda i, m: (result.append(("error", i, m)), done.set()),
    )
    done.wait(wait)
    return tokens, result


failures = 0


def check(name, cond):
    global failures
    print(("PASS " if cond else "FAIL ") + name)
    failures += not cond


srv, url = start_server()

# 1. normal streaming
eng = LLMEngine(url=url)
tokens, result = collect(eng, "What is Python")
text = "".join(t for _, t in tokens)
check("streams tokens in order", text == "ANSWER:What is Python end ")
check("reports done", result and result[0][0] == "done")

# 2. server check
check("check_server ok", eng.check_server()[0])

# 3. rapid fire: second question cancels the first, no stale tokens
eng = LLMEngine(url=url)
first_tokens, second_tokens, done = [], [], threading.Event()
id1 = eng.start_answer("slow one two three four five six", lambda i, t: first_tokens.append((i, t)),
                       lambda i: None, lambda i, m: None)
time.sleep(0.6)
id2 = eng.start_answer("fast", lambda i, t: second_tokens.append((i, t)),
                       lambda i: done.set(), lambda i, m: done.set())
done.wait(5)
time.sleep(0.8)  # give a zombie first job time to misbehave
late_first = [t for i, t in first_tokens if i == id1]
check("second answer completes", "".join(t for _, t in second_tokens) == "ANSWER:fast end ")
check("first job stopped after cancel (<= 4 tokens)", len(late_first) <= 4)
check("no first-job token arrived after second started streaming",
      all(i == id2 for i, _ in second_tokens))

# 4. HTTP error
eng = LLMEngine(model="missing", url=url)
tokens, result = collect(eng, "hello")
check("HTTP 404 reported as error", result and result[0][0] == "error" and "404" in result[0][2])

# 5. server down
eng = LLMEngine(url="http://127.0.0.1:9/api/generate", connect_timeout=1)
tokens, result = collect(eng, "hello")
check("connection refused reported", result and result[0][0] == "error")
check("check_server reports down", not eng.check_server()[0])

# 6. classification
cases = {
    "Who invented Python?": "fact",
    "What is AI?": "definition",
    "Explain neural networks.": "explain",
    "What is the difference between AI and ML?": "compare",
    "Difference between AI and ML?": "compare",
    "What is 25 multiplied by 4?": "math",
    "How does a neural network work?": "explain",
    "What is the Periodic Table?": "definition",
}
for q, style in cases.items():
    check("classify %r -> %s" % (q, style), classify_question(q) == style)

srv.shutdown()
print("ALL PASSED" if not failures else "%d FAILED" % failures)
sys.exit(1 if failures else 0)
#!/usr/bin/env python3
import json, logging, os, re, time, threading, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
tasks: dict[str, dict] = {}

class OpenAIClient:
    def __init__(self, model, api_key, base_url=None):
        from openai import OpenAI
        kwargs = {"api_key": api_key, "base_url": base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1/"}
        self.client = OpenAI(**kwargs)
        self.model = model
    def chat_completions_create(self, messages):
        header = {"x-DashScope-DataInspection": '{"input": "disable", "output": "disable"}'}
        r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.0, max_tokens=8192, extra_headers=header)
        return {"choices": [{"message": {"role": r.choices[0].message.role, "content": r.choices[0].message.content}, "finish_reason": r.choices[0].finish_reason}]}

def detect_reply_columns(columns):
    pattern = re.compile(r"^回复(\d+)$")
    cols = [(int(m.group(1)), c) for c in columns if (m := pattern.match(str(c)))]
    return [c for _, c in sorted(cols)]

def render_prompt(tpl, system, history, r1, r2):
    return tpl.replace("{{system}}", system).replace("{{chat}}", history).replace("{{bot1}}", r1).replace("{{bot2}}", r2)

def extract_json(text):
    text = text.strip()
    for fn in [lambda t: json.loads(t),
               lambda t: json.loads(re.search(r"```(?:json)?\s*\n?(.*?)\n?```", t, re.DOTALL).group(1).strip()),
               lambda t: json.loads(re.search(r"\{.*\}", t, re.DOTALL).group(0))]:
        try: return fn(text)
        except: pass
    return {"rank": "", "analysis": text}

def evaluate_single(client, system, history, r1, r2, tpl, max_retries=3):
    messages = [{"role": "user", "content": render_prompt(tpl, system, history, r1, r2)}]
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat_completions_create(messages)
            return extract_json(resp["choices"][0]["message"]["content"])
        except Exception as e:
            logger.warning("第%d次失败: %s", attempt, e)
            if attempt < max_retries: time.sleep(attempt)
            else: return {"rank": "", "analysis": str(e)}

def run_task(task_id, input_path, prompt_tpl, model, api_key, base_url, concurrency):
    try:
        tasks[task_id].update({"status": "running", "progress": 0, "message": "正在读取文件..."})
        df = pd.read_excel(input_path).fillna("")
        reply_cols = detect_reply_columns(list(df.columns))
        if len(reply_cols) < 2:
            tasks[task_id].update({"status": "error", "message": f"未找到足够回复列，识别到：{reply_cols}"}); return
        pairs = list(combinations(reply_cols, 2))
        total = len(df) * len(pairs)
        tasks[task_id]["message"] = f"识别到{len(reply_cols)}个回复列，{len(pairs)}个对比组合，共{total}个任务"
        tasks[task_id]["total"] = total
        client = OpenAIClient(model, api_key, base_url or None)
        pair_results = {f"{a}vs{b}": [{} for _ in range(len(df))] for a, b in pairs}
        completed = 0
        futures = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for idx, row in df.iterrows():
                sys_v, hist_v = str(row.get("System","")), str(row.get("对话历史",""))
                for a, b in pairs:
                    f = ex.submit(evaluate_single, client, sys_v, hist_v, str(row[a]), str(row[b]), prompt_tpl)
                    futures[f] = (idx, f"{a}vs{b}")
            for f in as_completed(futures):
                idx, pk = futures[f]
                try: pair_results[pk][idx].update(f.result())
                except Exception as e: logger.error("异常 行=%d 组合=%s: %s", idx, pk, e)
                completed += 1
                pct = int(completed / total * 100)
                tasks[task_id].update({"progress": pct, "message": f"评测进度：{completed}/{total}（{pct}%）"})
        out_df = df.reset_index(drop=True).copy()
        for a, b in pairs:
            pk = f"{a}vs{b}"
            na, nb = re.search(r"\d+", a).group(), re.search(r"\d+", b).group()
            out_df[f"{na}vs{nb}_rank"] = [r.get("rank","") for r in pair_results[pk]]
            out_df[f"{na}vs{nb}_analysis"] = [r.get("analysis","") for r in pair_results[pk]]
        out_path = os.path.join(OUTPUT_DIR, f"{task_id}_result.xlsx")
        out_df.to_excel(out_path, index=False, engine="openpyxl")
        tasks[task_id].update({"status": "done", "progress": 100, "message": "评测完成！", "output_file": f"{task_id}_result.xlsx"})
    except Exception as e:
        logger.exception("任务%s异常", task_id)
        tasks[task_id].update({"status": "error", "message": f"发生错误：{e}"})

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start_task():
    model = request.form.get("model","").strip()
    api_key = request.form.get("api_key","").strip()
    base_url = request.form.get("base_url","").strip()
    concurrency = int(request.form.get("concurrency", 5))
    prompt_text = request.form.get("prompt_text","").strip()
    input_file = request.files.get("input_file")
    prompt_file = request.files.get("prompt_file")
    if not model: return jsonify({"error": "请填写模型名称"}), 400
    if not api_key: return jsonify({"error": "请填写 API Key"}), 400
    if not input_file: return jsonify({"error": "请上传数据文件"}), 400
    if prompt_file and prompt_file.filename:
        p = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_prompt.xlsx")
        prompt_file.save(p)
        try:
            pdf = pd.read_excel(p)
            if "prompt" not in pdf.columns: return jsonify({"error": "提示词文件必须包含 prompt 列"}), 400
            prompt_tpl = str(pdf.iloc[0]["prompt"])
        except Exception as e: return jsonify({"error": f"读取提示词文件失败：{e}"}), 400
    elif prompt_text:
        prompt_tpl = prompt_text
    else:
        return jsonify({"error": "请上传提示词文件或粘贴提示词"}), 400
    task_id = str(uuid.uuid4())
    inp = os.path.join(UPLOAD_DIR, f"{task_id}_input.xlsx")
    input_file.save(inp)
    tasks[task_id] = {"status": "pending", "progress": 0, "message": "任务已创建...", "output_file": None}
    threading.Thread(target=run_task, args=(task_id, inp, prompt_tpl, model, api_key, base_url, concurrency), daemon=True).start()
    return jsonify({"task_id": task_id})

@app.route("/api/status/<task_id>")
def task_status(task_id):
    if task_id not in tasks: return jsonify({"error": "任务不存在"}), 404
    return jsonify(tasks[task_id])

@app.route("/api/download/<task_id>")
def download(task_id):
    if task_id not in tasks: return jsonify({"error": "任务不存在"}), 404
    t = tasks[task_id]
    if t["status"] != "done": return jsonify({"error": "文件尚未生成"}), 400
    return send_file(os.path.join(OUTPUT_DIR, t["output_file"]), as_attachment=True, download_name="eval_result.xlsx")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n✅ 网站已启动，端口：{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)

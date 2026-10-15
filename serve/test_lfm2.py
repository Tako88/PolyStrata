"""LFM2.5-24B-A2B on the Python side: its `lfm2` pre-tokenizer (the LLaMA-3 pattern under another name), its
reasoning (a `<think>` the template opens, so the parser starts inside it), and its Python-style tool calls
`<|tool_call_start|>[NAME(a=1), NAME2(b='x')]<|tool_call_end|>` (parallel calls comma-separated in one block).

    python -m unittest serve.test_lfm2
"""
import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser, parse_lfm2_tool_calls  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, call_format_of, serve  # noqa: E402

CTX = 4096


class PreTokenizer(unittest.TestCase):
    TOKENS = ["1", "2", "3", "4", "5", "12", "123", "45", "a", "b", "ab"]
    MERGES = ["1 2", "12 3", "4 5"]

    def pieces(self, text):
        tok = ST.Tokenizer(self.TOKENS, self.MERGES, pre="lfm2")
        return [self.TOKENS[i] for i in tok.encode(text)]

    def test_lfm2_is_the_llama3_pattern_and_ignores_merges(self):
        # llama.cpp's pre == "lfm2" selects LLAMA_VOCAB_PRE_TYPE_LLAMA3 (the same text as ChatGLM4) + ignore_merges
        self.assertIs(ST.PRE_PATTERNS["lfm2"], ST.GLM4_PATTERN)
        self.assertEqual(ST.LLAMA3_PATTERN, ST.GLM4_PATTERN)
        self.assertIn("lfm2", ST.IGNORE_MERGES)
        tok = ST.Tokenizer(self.TOKENS, self.MERGES, pre="lfm2")
        self.assertTrue(tok.ignore_merges)
        self.assertEqual(tok.pre_pattern, ST.LLAMA3_PATTERN)

    def test_digits_in_runs_of_three_and_no_merges(self):
        self.assertEqual(self.pieces("12345"), ["123", "45"])
        self.assertEqual(self.pieces("ab"), ["ab"])          # the whole piece is the token, no merges run


SCHEMAS = {t["name"]: t for t in [
    {"name": "special_function", "parameters": {"properties": {"arg1": {"type": "integer"}}}},
    {"name": "special_function_with_opt", "parameters": {"properties": {"arg1": {"type": "integer"},
                                                                        "arg2": {"type": "integer"}}}},
    {"name": "get_time", "parameters": {"properties": {"city": {"type": "string"}}}},
    {"name": "toggle", "parameters": {"properties": {"enabled": {"type": "boolean"}}}},
    {"name": "set_nullable", "parameters": {"properties": {"value": {"type": "null"}}}},
    {"name": "set_config", "parameters": {"properties": {"config": {"type": "object"}}}},
    {"name": "Calendar.create_event", "parameters": {"properties": {"title": {"type": "string"},
                                                                    "participants": {"type": "array"},
                                                                    "metadata": {"type": "object"}}}},
    {"name": "python", "parameters": {"properties": {"code": {"type": "string"}}}},
    {"name": "empty_args", "parameters": {"properties": {}}},
]}
TOOLS = [SCHEMAS[k] for k in SCHEMAS]


class Notation(unittest.TestCase):
    """The leaf forms of the block, fixture by fixture from llama.cpp tests/test-chat.cpp:1979-2075."""

    def parse(self, body):
        return [(c.name, c.arguments) for c in parse_lfm2_tool_calls(body, SCHEMAS)]

    def test_the_fixtures(self):
        cases = [
            ("[special_function(arg1=1)]", [("special_function", {"arg1": 1})]),
            ("[get_time(city=\"XYZCITY\")]", [("get_time", {"city": "XYZCITY"})]),
            ("[toggle(enabled=True)]", [("toggle", {"enabled": True})]),                 # Python literal
            ("[set_nullable(value=None)]", [("set_nullable", {"value": None})]),
            ("[set_config(config={\"enabled\": True, \"count\": 3})]",
             [("set_config", {"config": {"enabled": True, "count": 3}})]),              # nested Python
            ("[set_config(config={\"enabled\": true, \"note\": null})]",
             [("set_config", {"config": {"enabled": True, "note": None}})]),            # JSON literals too
            ("[Calendar.create_event(title=\"demo\", participants=[\"Alice\", \"Bob\"], "
             "metadata={\"priority\": \"high\", \"reminder\": true})]",
             [("Calendar.create_event", {"title": "demo", "participants": ["Alice", "Bob"],
                                         "metadata": {"priority": "high", "reminder": True}})]),  # dotted name
            ("[python(code=\"def hello():\\n    print('hey')\")]",
             [("python", {"code": "def hello():\n    print('hey')"})]),                # a real newline, not \n
            ("[python(code=\"First line\\nSecond line\\tindented\")]",
             [("python", {"code": "First line\nSecond line\tindented"})]),
            ("[python(code=\"print(\\\"hi\\\")\")]", [("python", {"code": 'print("hi")'})]),
            ("[empty_args()]", [("empty_args", {})]),
            ("[special_function(arg1=1), special_function_with_opt(arg1=1, arg2=2)]",
             [("special_function", {"arg1": 1}), ("special_function_with_opt", {"arg1": 1, "arg2": 2})]),
        ]
        for body, want in cases:
            with self.subTest(body=body):
                self.assertEqual(self.parse(body), want)

    def test_a_string_parameter_is_kept_as_text(self):
        # a bare string param is already text; a non-string literal on a string param is JSON-encoded, as
        # parse_gemma_tool_call does
        self.assertEqual(self.parse("[get_time(city=Paris)]"), [("get_time", {"city": "Paris"})])
        self.assertEqual(self.parse("[get_time(city=7)]"), [("get_time", {"city": "7"})])
        self.assertEqual(self.parse("[special_function(arg1=7)]"), [("special_function", {"arg1": 7})])


class Parser(unittest.TestCase):
    SCHEMA = [SCHEMAS["get_time"], SCHEMAS["special_function"], SCHEMAS["special_function_with_opt"],
              SCHEMAS["python"]]

    def events(self, text, stream_tools=False, step=1, thinking=False, opened=False):
        p = OutputParser(thinking=thinking, tools=self.SCHEMA, stream_tools=stream_tools, call_format="lfm2",
                         opened=opened)
        return [e for i in range(0, len(text), step) for e in p.feed(text[i:i + step])] + p.finish()

    def calls(self, evs):
        return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]

    def test_an_answer_without_a_call_is_content(self):
        text = "Hello, world!\nWhat's up?"
        evs = self.events(text, thinking=False)
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), text)
        self.assertEqual([e.kind for e in evs if e.kind != "content"], [])

    def test_text_before_a_call_stays_content(self):
        text = "Let me check the time.<|tool_call_start|>[get_time(city=\"Paris\")]<|tool_call_end|>"
        for step in (1, 5, 10_000):
            evs = self.events(text, step=step, thinking=False)
            self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Let me check the time.")
            self.assertEqual(self.calls(evs), [("get_time", {"city": "Paris"})])

    def test_a_call_in_any_pieces(self):
        text = "<|tool_call_start|>[get_time(city=\"Paris\")]<|tool_call_end|>"
        for step in (1, 3, 7, len(text)):
            with self.subTest(step=step):
                self.assertEqual(self.calls(self.events(text, step=step, thinking=False)),
                                 [("get_time", {"city": "Paris"})])

    def test_two_calls_in_one_block(self):
        text = ("<|tool_call_start|>[special_function(arg1=1), special_function_with_opt(arg1=1, arg2=2)]"
                "<|tool_call_end|>")
        self.assertEqual(self.calls(self.events(text, step=1, thinking=False)),
                         [("special_function", {"arg1": 1}), ("special_function_with_opt", {"arg1": 1, "arg2": 2})])

    def test_a_streamed_block_announces_its_first_call_once(self):
        text = "<|tool_call_start|>[special_function(arg1=1)]<|tool_call_end|>"
        evs = [e for e in self.events(text, stream_tools=True, step=1, thinking=False) if e.kind.startswith("tool_")]
        self.assertEqual([e.kind for e in evs], ["tool_start", "tool_args", "tool_call"])
        self.assertEqual(json.loads(evs[1].text), {"arg1": 1})
        self.assertEqual({e.call.id for e in evs}, {evs[0].call.id})

    def test_a_partial_block_waits_for_its_terminator(self):
        # the block is held whole: its Python-ish values are not JSON, so they cannot be streamed as JSON pieces
        p = OutputParser(thinking=False, tools=self.SCHEMA, call_format="lfm2")
        self.assertEqual(p.feed("<|tool_call_start|>[get_time(city="), [])
        self.assertEqual(p.feed("\"Paris\""), [])
        evs = p.feed(")]<|tool_call_end|>") + p.finish()
        self.assertEqual(self.calls(evs), [("get_time", {"city": "Paris"})])

    def test_a_call_the_output_ended_in_is_text(self):
        text = "<|tool_call_start|>[get_time(city=\"Par"
        evs = self.events(text, thinking=False)
        self.assertEqual([(e.kind, e.text) for e in evs], [("content", text)])

    def test_a_marker_inside_a_value_does_not_end_the_block(self):
        # lfm2_call_end is quote-aware: the marker text inside a string is part of it, as gemma_call_end is
        text = ("<|tool_call_start|>[python(code=\"x = '<|tool_call_end|>'\")]<|tool_call_end|>")
        self.assertEqual(self.calls(self.events(text, step=3, thinking=False)),
                         [("python", {"code": "x = '<|tool_call_end|>'"})])

    def test_a_parallel_streamed_block_streams_the_first_call_then_both(self):
        # the block parses whole (its values are Python-ish), so only the first call's arguments stream; both
        # calls are then delivered, the first reusing the streamed id
        text = ("<|tool_call_start|>[special_function(arg1=1), special_function_with_opt(arg1=1, arg2=2)]"
                "<|tool_call_end|>")
        evs = [e for e in self.events(text, stream_tools=True, step=1, thinking=False)
               if e.kind.startswith("tool_")]
        self.assertEqual([e.kind for e in evs], ["tool_start", "tool_args", "tool_call", "tool_call"])
        self.assertEqual(json.loads(evs[1].text), {"arg1": 1})
        self.assertEqual(evs[0].call.id, evs[2].call.id)
        self.assertNotEqual(evs[0].call.id, evs[3].call.id)
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs[2:]],
                         [("special_function", {"arg1": 1}),
                          ("special_function_with_opt", {"arg1": 1, "arg2": 2})])

    def test_a_malformed_complete_block_becomes_content_never_an_error(self):
        # a body with no "(" (parse_lfm2_tool_calls raises), an empty [] block (no calls), and a call with an
        # empty name: each surfaces as its own text out of feed(), as an unterminated block already does in
        # finish()
        for body in ("just prose, no call here", "[]", "[(a=1)]"):
            text = f"<|tool_call_start|>{body}<|tool_call_end|>"
            for step in (1, 4, 10_000):
                with self.subTest(body=body, step=step):
                    self.assertEqual([(e.kind, e.text) for e in self.events(text, step=step, thinking=False)],
                                     [("content", text)])
            self.events(text, stream_tools=True, step=1, thinking=False)   # must not raise either way


class Reasoning(unittest.TestCase):
    def events(self, text, thinking=True, opened=False):
        p = OutputParser(thinking=thinking, call_format="lfm2", opened=opened)
        return p.feed(text) + p.finish()

    def test_a_prompt_that_opened_the_think_starts_inside_it(self):
        evs = self.events("The user asks the capital.</think>\n\nParis.", opened=True)
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "The user asks the capital.")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Paris.")

    def test_a_reasoning_that_ends_on_a_quote_still_ends(self):
        # LFM2.5 writes </think> and the answer at once (`"Paris"</think>Paris`, measured in Step 22): unlike the
        # ChatML form, a closing quote before the tag does not make it a quoted one
        evs = self.events('so I will say "Paris"</think>Paris', opened=True)
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), 'so I will say "Paris"')
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Paris")

    def test_the_models_own_think_opener_is_stripped(self):
        # thinking off (the prompt closed <think>): a model that opens one anyway has it stripped
        evs = self.events("<think>The user asks.</think>\n\nParis.", thinking=False)
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "The user asks.")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Paris.")
        # and with the same text, without the opener, it is content
        evs = self.events("Paris.", thinking=False)
        self.assertEqual([(e.kind, e.text) for e in evs], [("content", "Paris.")])


class CallFormat(unittest.TestCase):
    def test_every_pre(self):
        self.assertEqual(call_format_of("glm4"), "glm")
        self.assertEqual(call_format_of("glm5"), "glm")
        self.assertEqual(call_format_of("gemma4"), "gemma")
        self.assertEqual(call_format_of("lfm2"), "lfm2")
        self.assertEqual(call_format_of("qwen35"), "function")
        self.assertEqual(call_format_of(""), "function")


class GenerationTags(unittest.TestCase):
    """LFM2.5's GGUF template tags the assistant span with transformers' `{% generation %}` / `{% endgeneration %}`
    markers; the extension must parse them and emit nothing (llama.cpp's minja ignores them the same way)."""

    def test_the_generation_markers_emit_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text("A{%- generation -%}B{{ messages[0]['content'] }}C{%- endgeneration -%}D", encoding="utf-8")
            tpl = ChatTemplate(path, format="lfm2")
        self.assertEqual(tpl.render([{"role": "user", "content": "X"}], add_generation_prompt=False), "ABXCD")


# The relevant shape of the 24B GGUF template: a system turn with the tools as JSON, ChatML turns with LFM2 tool
# calls for the assistant, and a generation prompt that opens <think>.
LFM2_TEMPLATE = r"""{{- bos_token -}}
{%- set ns = namespace(system="") -%}
{%- if messages and messages[0]["role"] == "system" -%}
{%- set ns.system = messages[0]["content"] -%}
{%- set messages = messages[1:] -%}
{%- endif -%}
{%- if tools -%}
{%- set ns.system = ns.system + ("\n" if ns.system else "") + "List of tools: " + (tools | tojson) -%}
{%- endif -%}
{%- if ns.system -%}
{{- "<|im_start|>system\n" + ns.system + "<|im_end|>\n" -}}
{%- endif -%}
{%- for m in messages -%}
{{- "<|im_start|>" + m["role"] + "\n" -}}
{%- if m["role"] == "assistant" -%}
{%- set thinking = m.get("reasoning_content") -%}
{%- if thinking -%}{{- "<think>" + thinking + "</think>" -}}{%- endif -%}
{{- m["content"] -}}
{%- for c in m.get("tool_calls") or [] -%}
{%- set fn = c["function"] if "function" in c else c -%}
{{- "<|tool_call_start|>[" + fn["name"] + "(" -}}
{%- for k, v in (fn.get("arguments") or {}).items() -%}
{%- if not loop.first -%}{{- ", " -}}{%- endif -%}
{{- k + "=" -}}{%- if v is string -%}{{- "'" + v + "'" -}}{%- else -%}{{- v | tojson -}}{%- endif -%}
{%- endfor -%}
{{- ")]<|tool_call_end|>" -}}
{%- endfor -%}
{%- else -%}
{{- m.get("content") or "" -}}
{%- endif -%}
{{- "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- "<|im_start|>assistant\n<think>" -}}
{%- endif -%}"""


class Lfm2Tokens(ByteTokenizer):
    SPECIALS = ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
                "<|tool_call_start|>", "<|tool_call_end|>", "<|tool_list_start|>", "<|tool_list_end|>"]
    pre = "lfm2"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 258}


class CapsProbe(unittest.TestCase):
    def test_the_probe_reads_the_lfm2_call_notation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(LFM2_TEMPLATE, encoding="utf-8")
            caps = ChatTemplate(path, format="lfm2").caps
        self.assertTrue(caps["supports_tools"])
        self.assertTrue(caps["supports_tool_calls"])
        self.assertTrue(caps["supports_parallel_tool_calls"])


class ServiceForLfm2(unittest.TestCase):
    """A scripted request through the real Service and the real HTTP handlers, both APIs, whole and streamed."""
    TOOLS = {
        "openai": [{"type": "function", "function": {"name": "get_time", "description": "Get the time in a city",
                     "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                    "required": ["city"]}}}],
        "anthropic": [{"name": "get_time", "description": "Get the time in a city",
                       "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                        "required": ["city"]}}],
    }

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.tpl = Path(self.dir.name) / "chat_template.jinja"
        self.tpl.write_text(LFM2_TEMPLATE, encoding="utf-8")
        self.tok = Lfm2Tokens()

    def run_requests(self, script, req_extra=None):
        """-> {("openai"|"anthropic", stream): (stop reason, [arguments])} for the model's `script`."""
        svc = Service(MockEngine(self.tok, script, max_context=CTX), self.tok, ChatTemplate(self.tpl, format="lfm2"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = {}
        try:
            for api, path in (("openai", "/v1/chat/completions"), ("anthropic", "/v1/messages")):
                for stream in (False, True):
                    body = {"model": "x", "max_tokens": 500, "stream": stream, "tools": self.TOOLS[api],
                            "messages": [{"role": "user", "content": "What time is it?"}]}
                    if api == "openai":
                        body.update(req_extra or {})
                    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={
                        "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        raw = r.read().decode()
                    if api == "openai" and not stream:
                        c = json.loads(raw)["choices"][0]
                        out[api, stream] = (c["finish_reason"], [json.loads(tc["function"]["arguments"])
                                                                 for tc in c["message"].get("tool_calls") or []])
                    elif api == "openai":
                        evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                        args = {}
                        for e in evs:
                            for tc in e["choices"][0]["delta"].get("tool_calls") or []:
                                i = tc.get("index", 0)
                                args[i] = args.get(i, "") + ((tc.get("function") or {}).get("arguments") or "")
                        out[api, stream] = (evs[-1]["choices"][0]["finish_reason"],
                                            [json.loads(args[i]) for i in sorted(args)])
                    elif not stream:
                        m = json.loads(raw)
                        out[api, stream] = (m["stop_reason"],
                                            [b["input"] for b in m["content"] if b["type"] == "tool_use"])
                    else:
                        evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                        args = {}
                        for e in evs:
                            if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
                                args[e["index"]] = args.get(e["index"], "") + e["delta"]["partial_json"]
                        stop = next(e["delta"]["stop_reason"] for e in evs if e["type"] == "message_delta")
                        out[api, stream] = (stop, [json.loads(args[i]) for i in sorted(args)])
        finally:
            httpd.shutdown()
            httpd.server_close()
        return out

    def test_a_single_call_over_both_apis_whole_and_streamed(self):
        got = self.run_requests("</think>\n\n<|tool_call_start|>[get_time(city='Paris')]<|tool_call_end|>")
        self.assertEqual(got, {
            ("openai", False): ("tool_calls", [{"city": "Paris"}]),
            ("openai", True): ("tool_calls", [{"city": "Paris"}]),
            ("anthropic", False): ("tool_use", [{"city": "Paris"}]),
            ("anthropic", True): ("tool_use", [{"city": "Paris"}]),
        })

    def test_a_parallel_call_over_both_apis_whole_and_streamed(self):
        got = self.run_requests("</think>\n\n<|tool_call_start|>[get_time(city='Paris'), get_time(city='Lyon')]"
                                "<|tool_call_end|>")
        self.assertEqual(got, {
            ("openai", False): ("tool_calls", [{"city": "Paris"}, {"city": "Lyon"}]),
            ("openai", True): ("tool_calls", [{"city": "Paris"}, {"city": "Lyon"}]),
            ("anthropic", False): ("tool_use", [{"city": "Paris"}, {"city": "Lyon"}]),
            ("anthropic", True): ("tool_use", [{"city": "Paris"}, {"city": "Lyon"}]),
        })

    def test_a_plain_answer_with_thinking_off(self):
        # enable_thinking false: the template's <think> is closed and the answer is content, not reasoning
        body = {"model": "x", "max_tokens": 64, "messages": [{"role": "user", "content": "Capital of France?"}],
                "chat_template_kwargs": {"enable_thinking": False}}
        svc = Service(MockEngine(self.tok, "Paris.", max_context=CTX), self.tok,
                      ChatTemplate(self.tpl, format="lfm2"))
        httpd = serve(svc, port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                msg = json.loads(r.read().decode())["choices"][0]["message"]
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(msg["content"], "Paris.")
        self.assertFalse(msg.get("reasoning_content"))

    def test_a_thinking_answer_goes_in_reasoning_content(self):
        script = "The user asks the capital.</think>\n\nParis."
        svc = Service(MockEngine(self.tok, script, max_context=CTX), self.tok,
                      ChatTemplate(self.tpl, format="lfm2"))
        httpd = serve(svc, port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps({"model": "x", "max_tokens": 64,
                                                          "messages": [{"role": "user", "content": "Capital?"}]}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                msg = json.loads(r.read().decode())["choices"][0]["message"]
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(msg["reasoning_content"], "The user asks the capital.")
        self.assertEqual(msg["content"], "Paris.")


if __name__ == "__main__":
    unittest.main()

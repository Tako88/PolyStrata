"""Nemotron-3.5-Bolt-74B-A9B behind the server: its pixtral pre-tokenizer (llama.cpp's LLAMA3 arm), its ChatML
template's `<tool_call>` grammar, thinking, the capability hints, and a scripted run through the real Service.

The template the GGUF carries is written to the pack's tokenizer directory by tools/strata_tokenizer.py; this file
does not author it.  The reduced TEMPLATE below keeps the parts the server meets - the system/tool/user/assistant/tool
turns, the tool instructions, the call grammar, and the reasoning the template rewrites to `<think></think>` in a
finished turn - and the reference's own rendered prompt was compared string for string in evidence.md (M7, step 24).

    python -m unittest serve.test_nemotron
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

from serve.frontend import ChatTemplate, OutputParser, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

CTX = 4096


def call(name: str, params) -> str:
    """One call exactly as the template's instructions write it: `<tool_call>`, `<function=NAME>`, one `<parameter=P>`
    a line of the value, `</function>`, `</tool_call>`."""
    body = f"<tool_call>\n<function={name}>\n"
    for key, value in params:
        body += f"<parameter={key}>\n{value}\n</parameter>\n"
    return body + "</function>\n</tool_call>"


SCHEMA = [
    {"name": "write", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "n": {"type": "integer"}, "flag": {"type": "boolean"},
        "nothing": {"type": "null"}, "meta": {"type": "object"}, "tags": {"type": "array"}}}},
    {"name": "now", "parameters": {"type": "object", "properties": {}}},
]
TOOLS = [{"type": "function", "function": t} for t in SCHEMA]

# The reference /apply-template answer saved by M6 step 19 for the request in test_the_rendered_prompt_matches_the_reference
# (.pi/plans/nemotron/m6/step19-apply-template.txt): the reduced TEMPLATE must render it string for string.
APPLY_TEMPLATE = '<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>get_weather</name>\n<description>Get the current weather for a city</description>\n<parameters>\n<parameter>\n<name>city</name>\n<type>string</type>\n<description>The city name</description>\n</parameter>\n<required>["city"]</required>\n</parameters>\n</function>\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n'


def events(text: str, tools=None, thinking=True, stream_tools=False, step=10_000, recover=False):
    """Feed `text` to the function-format parser and finish it, the way the server runs a reply."""
    p = OutputParser(thinking=thinking, tools=tools if tools is not None else SCHEMA, stream_tools=stream_tools,
                     recover=recover, call_format="function")
    out = []
    for i in range(0, len(text), step):
        out += p.feed(text[i:i + step])
    return out + p.finish()


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def joined(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class Parsing(unittest.TestCase):
    """The template's grammar, read whole and in pieces: a reply starts inside the reasoning the prompt opened and
    ends it at `</think>`; a call is `<tool_call>` then (after whitespace) `<function=`."""

    ONE = call("write", [("path", "doc.md"), ("n", "3")])

    def test_content_only(self):
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = events("The capital of France is Paris.", tools=[], thinking=False, step=step)
                self.assertEqual(joined(evs, "content"), "The capital of France is Paris.")
                self.assertEqual(calls(evs), [])

    def test_one_call(self):
        evs = events("plan</think>\n\n" + self.ONE)
        self.assertEqual(joined(evs, "reasoning"), "plan")
        self.assertEqual(calls(evs), [("write", {"path": "doc.md", "n": 3})])

    def test_a_string_argument_keeps_its_text(self):
        # the template writes a string raw, so a value may hold quotes and span lines
        text = call("write", [("path", 'a "b"\nc.txt')])
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                self.assertEqual(calls(events("x</think>\n\n" + text, step=step)),
                                 [("write", {"path": 'a "b"\nc.txt'})])

    def test_numbers_bools_and_null_are_typed(self):
        text = call("write", [("n", "42"), ("flag", "true"), ("nothing", "null")])
        self.assertEqual(calls(events("x</think>\n\n" + text)),
                         [("write", {"n": 42, "flag": True, "nothing": None})])

    def test_a_nested_object_and_a_list(self):
        text = call("write", [("meta", '{"a": 1, "b": [true, null]}'), ("tags", '[1, 2, "x"]')])
        self.assertEqual(calls(events("x</think>\n\n" + text)),
                         [("write", {"meta": {"a": 1, "b": [True, None]}, "tags": [1, 2, "x"]})])

    def test_two_calls_in_a_row(self):
        # the template writes one <function=...> per <tool_call>, so a parallel answer is two blocks in a row
        text = "x</think>\n\n" + call("write", [("path", "a.md")]) + "\n" + call("now", [])
        self.assertEqual(calls(events(text)), [("write", {"path": "a.md"}), ("now", {})])

    def test_text_before_the_call(self):
        evs = events("x</think>\n\nI will write it.\n" + self.ONE)
        self.assertEqual(joined(evs, "content"), "I will write it.")
        self.assertEqual(calls(evs), [("write", {"path": "doc.md", "n": 3})])

    def test_a_partial_call_is_held_then_completes(self):
        p = OutputParser(thinking=True, tools=SCHEMA, call_format="function")
        evs = p.feed("plan</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\ndoc.md")
        self.assertEqual([e.kind for e in evs], ["reasoning"])
        evs += p.feed("\n</parameter>\n</function>\n</tool_call>") + p.finish()
        self.assertEqual(calls(evs), [("write", {"path": "doc.md"})])
        self.assertEqual(joined(evs, "content"), "")

    def test_an_unterminated_call_is_text_at_finish(self):
        text = "plan</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\ndoc.md\n</parameter>\n"
        evs = events(text)
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=write>", joined(evs, "content"))

    def test_the_think_opener_is_already_consumed(self):
        # the prompt ended on <|im_start|>assistant\n<think>\n: the reply's own first text is reasoning
        evs = events("I should think about it.\n</think>\n\nThe answer is 4.")
        self.assertEqual(joined(evs, "reasoning"), "I should think about it.\n")
        self.assertEqual(joined(evs, "content"), "The answer is 4.")

    def test_thinking_off_starts_in_the_content(self):
        # the prompt ended on <|im_start|>assistant\n<think></think>: newlines before the answer are dropped
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = events("\nThe answer is 4.", thinking=False, step=step)
                self.assertEqual(joined(evs, "reasoning"), "")
                self.assertEqual(joined(evs, "content"), "The answer is 4.")

    def test_a_call_is_streamed_whole_and_in_pieces(self):
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = events("x</think>\n\n" + self.ONE, stream_tools=True, step=step)
                self.assertEqual(calls(evs), [("write", {"path": "doc.md", "n": 3})])
                if step < 10_000:
                    self.assertEqual(json.loads(joined(evs, "tool_args")), {"path": "doc.md", "n": 3})
                    starts = [e for e in evs if e.kind == "tool_start"]
                    self.assertEqual([e.call.name for e in starts], ["write"])


# The parts of the GGUF's ChatML template the server has to meet: the system turn with the tool list, the
# <|im_start|>/<|im_end|> turns, calls as <tool_call><function=NAME>, results as <tool_response>, the tool
# instructions naming <tool_call> and <function=, a finished turn's reasoning rewritten to <think></think>, and an
# assistant turn that opens <think> when thinking is on or <think></think> when it is off.
TEMPLATE = """{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}
{%- set truncate_history_thinking = truncate_history_thinking if truncate_history_thinking is defined else True %}
{%- if messages[0]["role"] == "system" %}{%- set system_message = messages[0]["content"] %}{%- set loop_messages = messages[1:] %}
{%- else %}{%- set system_message = "" %}{%- set loop_messages = messages %}{%- endif %}
{%- if not tools %}{%- set tools = [] %}{%- endif %}
{%- set ns = namespace(last_user_idx = -1) %}
{%- for m in loop_messages %}{%- if m["role"] == "user" %}{%- set ns.last_user_idx = loop.index0 %}{%- endif %}{%- endfor %}
{%- if system_message is defined %}{{- "<|im_start|>system\\n" ~ system_message }}
{%- else %}{%- if tools | length > 0 %}{{- "<|im_start|>system\\n" }}{%- endif %}{%- endif %}
{%- if tools | length > 0 %}
{%- if system_message is defined and system_message | length > 0 %}{{- "\\n\\n" }}{%- endif %}
{{- "# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>" }}
{%- for tool in tools %}
{%- if tool.function is defined %}{%- set tool = tool.function %}{%- endif %}
{{- "\\n<function>\\n<name>" ~ tool.name ~ "</name>" }}
{%- if tool.description is defined %}{{- '\\n<description>' ~ (tool.description | trim) ~ '</description>' }}{%- endif %}
{{- '\\n<parameters>' }}
{%- if tool.parameters is defined and tool.parameters is mapping and tool.parameters.properties is defined and tool.parameters.properties is mapping %}
{%- for param_name, param_fields in tool.parameters.properties|items %}
{{- '\\n<parameter>\\n<name>' ~ param_name ~ '</name>' }}
{%- if param_fields.type is defined %}{{- '\\n<type>' ~ (param_fields.type | string) ~ '</type>' }}{%- endif %}
{%- if param_fields.description is defined %}{{- '\\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}{%- endif %}
{{- '\\n</parameter>' }}
{%- endfor %}{%- endif %}
{%- if tool.parameters is defined and tool.parameters.required is defined %}{{- '\\n<required>' ~ (tool.parameters.required | tojson | safe) ~ '</required>' }}{%- endif %}
{{- '\\n</parameters>\\n</function>' }}
{%- endfor %}
{{- "\\n</tools>" }}
{{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}
{%- endif %}
{%- if system_message is defined %}{{- '<|im_end|>\\n' }}{%- else %}{%- if tools | length > 0 %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}
{%- for message in loop_messages %}
{%- set content = message.content | default('', true) %}
{%- if message.role == "assistant" %}
{%- if message.reasoning_content is defined and message.reasoning_content is string and message.reasoning_content | trim | length > 0 %}
{%- set content = "<think>\\n" ~ message.reasoning_content ~ "</think>" ~ content %}
{%- elif '<think>' not in content and '</think>' not in content %}{%- set content = "<think></think>" ~ content %}{%- endif %}
{%- if message.tool_calls is defined and message.tool_calls | length > 0 %}
{{- '<|im_start|>assistant\\n' }}
{%- set include_content = not (truncate_history_thinking and loop.index0 < ns.last_user_idx) %}
{%- if content is string and content | trim | length > 0 %}
{%- if include_content %}{{- (content | trim) ~ '\\n' }}{%- else %}{%- set c = content | string %}{%- if '</think>' in c %}{%- set c = c.split('</think>')[-1] %}{%- elif '<think>' in c %}{%- set c = c.split('<think>')[0] %}{%- endif %}{{- "<think></think>" ~ c ~ '\\n' }}{%- endif %}
{%- else %}{{- "<think></think>" }}{%- endif %}
{%- for tool_call in message.tool_calls %}
{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}
{{- '<tool_call>\\n<function=' ~ tool_call.name ~ '>\\n' }}
{%- if tool_call.arguments is defined %}
{%- for args_name, args_value in tool_call.arguments|items %}
{{- '<parameter=' ~ args_name ~ '>\\n' }}
{%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
{{- args_value ~ '\\n</parameter>\\n' }}
{%- endfor %}{%- endif %}
{{- '</function>\\n</tool_call>\\n' }}
{%- endfor %}
{{- '<|im_end|>\\n' }}
{%- else %}
{%- if not (truncate_history_thinking and loop.index0 < ns.last_user_idx) %}
{{- '<|im_start|>assistant\\n' ~ (content | string | trim) ~ '<|im_end|>\\n' }}
{%- else %}
{%- set c = content | string %}{%- if '<think>' in c and '</think>' in c %}{%- set c = "<think></think>" ~ c.split('</think>')[-1] %}{%- endif %}
{%- set c = c | trim %}{% if c | length > 0 %}{{- '<|im_start|>assistant\\n' ~ c ~ '<|im_end|>\\n' }}{% else %}{{- '<|im_start|>assistant\\n<|im_end|>\\n' }}{% endif %}
{%- endif %}
{%- endif %}
{%- elif message.role == "user" or message.role == "system" %}
{{- '<|im_start|>' + message.role + '\\n' }}{{- content }}{{- '<|im_end|>\\n' }}
{%- elif message.role == "tool" %}
{%- if loop.previtem and loop.previtem.role != "tool" %}{{- '<|im_start|>user\\n' }}{%- endif %}
{{- '<tool_response>\\n' }}{{- content }}{{- '\\n</tool_response>\\n' }}
{%- if not loop.last and loop.nextitem.role != "tool" %}{{- '<|im_end|>\\n' }}{%- elif loop.last %}{{- '<|im_end|>\\n' }}{%- endif %}
{%- else %}{{- '<|im_start|>' ~ message.role ~ '\\n' ~ content ~ '<|im_end|>\\n' }}{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{%- if enable_thinking %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- else %}{{- '<|im_start|>assistant\\n<think></think>' }}{%- endif %}
{%- endif -%}"""


class NemotronTokens(ByteTokenizer):
    # the GGUF's own set: <|im_start|>/<|im_end|> are CONTROL, the tool tags USER_DEFINED, and there is no
    # <|endoftext|> in the vocabulary (eos_token_id 11 is <|im_end|>)
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|tool_call>", "</tool_call>",
                "<|tool_response>", "</tool_response>"]
    ALWAYS = ("<|tool_call>", "</tool_call>", "<|tool_response>", "</tool_response>")
    pre = "pixtral"
    special_ids = {"tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 11}


class ServiceForNemotron(unittest.TestCase):
    def setUp(self):
        self.tok = NemotronTokens()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "chat_template.jinja"
        self.path.write_text(TEMPLATE, encoding="utf-8")
        self.svc = Service(MockEngine(self.tok, "hi"), self.tok, ChatTemplate(self.path))

    def prompt(self, req):
        messages, tools, kwargs = openai_to_messages(req)
        return self.svc.render_prompt(messages, tools, kwargs)

    def test_call_format_turns_and_think_open_ids(self):
        self.assertEqual(self.svc.call_format, "function")
        self.assertEqual(self.svc.turn_ids, frozenset({self.tok.SPECIALS.index("<|im_start|>") + 256}))
        self.assertEqual(self.svc.think_open_ids, [])
        # the vocabulary has no <|endoftext|>: only <|im_end|> ends a reply, not the pieces of that literal
        self.assertEqual(self.svc.stop_ids, {self.tok.SPECIALS.index("<|im_end|>") + 256})
        self.assertNotIn(ord("<"), self.svc.stop_ids)

    def test_the_capabilities_of_the_template(self):
        # the rendered prompt carries the tool list and the call format; a finished turn's reasoning is dropped
        self.assertEqual(self.svc.template.caps, {"supports_tools": True, "supports_tool_calls": True,
                                                  "supports_system_role": True, "supports_parallel_tool_calls": True,
                                                  "supports_preserve_reasoning": False})

    def test_tools_and_thinking_in_the_prompt(self):
        text = self.prompt({"messages": [{"role": "system", "content": "You are a helpful assistant."},
                                         {"role": "user", "content": "What is the weather in Paris?"}],
                            "tools": TOOLS})
        self.assertTrue(text.startswith("<|im_start|>system\nYou are a helpful assistant.\n\n# Tools"))
        self.assertIn("<name>write</name>", text)
        self.assertIn("<tool_call>\n<function=example_function_name>", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n"))

    def test_the_rendered_prompt_matches_the_reference(self):
        # the request the reference's /apply-template got in M6 step 19, rendered back string for string
        tools = [{"type": "function", "function": {"name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {"type": "object", "properties": {
                        "city": {"type": "string", "description": "The city name"}}, "required": ["city"]}}}]
        text = self.prompt({"messages": [{"role": "system", "content": "You are a helpful assistant."},
                                         {"role": "user", "content": "What is the weather in Paris?"}],
                            "tools": tools})
        self.assertEqual(text, APPLY_TEMPLATE)

    def test_the_thinking_off_variant_closes_the_block(self):
        text = self.prompt({"messages": [{"role": "user", "content": "q"}], "tools": TOOLS,
                            "chat_template_kwargs": {"enable_thinking": False}})
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think></think>"))

    def test_a_call_in_the_history_is_rendered_in_the_template_grammar(self):
        messages = [{"role": "user", "content": "time?"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": "now", "arguments": "{}"}}]},
                    {"role": "tool", "content": "12:00"}]
        text = self.prompt({"messages": messages})
        self.assertIn("<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=now>\n</function>\n</tool_call>\n"
                      "<|im_end|>\n", text)
        self.assertIn("<|im_start|>user\n<tool_response>\n12:00\n</tool_response>\n<|im_end|>\n", text)

    def test_a_finished_turns_reasoning_is_dropped(self):
        text = self.prompt({"messages": [{"role": "user", "content": "2+2?"},
                                         {"role": "assistant", "content": "4", "reasoning_content": "private thoughts"},
                                         {"role": "user", "content": "and 3+3?"}]})
        self.assertNotIn("private thoughts", text)
        self.assertIn("<|im_start|>assistant\n<think></think>4<|im_end|>", text)


class OverHttp(unittest.TestCase):
    """One scripted run through the real Service: a plain answer, a thinking answer, a single and a parallel call,
    whole and streamed, with the mock engine standing in for strata-poly."""

    def setUp(self):
        self.tok = NemotronTokens()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "chat_template.jinja"
        self.path.write_text(TEMPLATE, encoding="utf-8")

    def start(self, script):
        svc = Service(MockEngine(self.tok, script, max_context=CTX), self.tok, ChatTemplate(self.path))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    @staticmethod
    def post(base, body, stream=False):
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        if not stream:
            return json.loads(raw)["choices"][0]
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]

    def test_a_plain_chat_answer(self):
        base = self.start("Bonjour !")
        choice = self.post(base, {"model": "m", "messages": [{"role": "user", "content": "Bonjour"}],
                                  "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(choice["message"]["content"], "Bonjour !")
        self.assertEqual(choice["finish_reason"], "stop")

    def test_a_thinking_answer_separates_reasoning_from_content(self):
        base = self.start("The user greets me.\n</think>\n\nBonjour !")
        choice = self.post(base, {"model": "m", "messages": [{"role": "user", "content": "Bonjour"}]})
        self.assertEqual(choice["message"]["content"], "Bonjour !")
        self.assertIn("The user greets me.", choice["message"]["reasoning_content"])
        self.assertNotIn("<think>", choice["message"]["content"] + choice["message"]["reasoning_content"])

    def test_a_single_call_whole_and_streamed(self):
        script = "I will check.\n</think>\n\n" + call("now", [])
        body = {"model": "m", "messages": [{"role": "user", "content": "time?"}], "tools": TOOLS}
        choice = self.post(self.start(script), body)
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"]))
                          for c in choice["message"]["tool_calls"]], [("now", {})])

        evs = self.post(self.start(script), {**body, "stream": True}, stream=True)
        self.assertEqual(evs[-1]["choices"][0]["finish_reason"], "tool_calls")
        streamed = "".join((tc.get("function") or {}).get("arguments") or ""
                           for e in evs for tc in e["choices"][0]["delta"].get("tool_calls") or [])
        self.assertEqual(json.loads(streamed), {})

    def test_two_calls_in_one_reply(self):
        script = ("Checking.\n</think>\n\n" + call("write", [("path", "a.md")]) + "\n" + call("now", []))
        choice = self.post(self.start(script),
                           {"model": "m", "messages": [{"role": "user", "content": "do it"}], "tools": TOOLS})
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"]))
                          for c in choice["message"]["tool_calls"]], [("write", {"path": "a.md"}), ("now", {})])

    def test_a_follow_up_after_a_tool_response(self):
        script = "12:00, then 12:01.\n</think>\n\nIt is 12:01."
        messages = [{"role": "user", "content": "time?"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": "now", "arguments": "{}"}}]},
                    {"role": "tool", "content": "12:00"},
                    {"role": "user", "content": "and now?"}]
        choice = self.post(self.start(script), {"model": "m", "messages": messages, "tools": TOOLS})
        self.assertEqual(choice["message"]["content"], "It is 12:01.")
        self.assertEqual(choice["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()

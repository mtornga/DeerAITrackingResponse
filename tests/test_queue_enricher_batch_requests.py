from importlib import util
from pathlib import Path


import sys
def _load_batch_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "agents" / "queue-enricher" / "batch_requests.py"
    spec = util.spec_from_file_location("batch_requests", module_path)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_build_vision_chat_body():
    mod = _load_batch_module()
    body = mod.build_vision_chat_body(
        model="gpt-4o-mini",
        prompt="Return JSON only.",
        image_urls=["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"],
    )
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"]["type"] == "json_object"
    assert body["messages"][0]["content"][0]["type"] == "text"
    assert body["messages"][0]["content"][1]["type"] == "image_url"
    assert body["messages"][0]["content"][1]["image_url"]["detail"] == "low"


def test_batch_request_jsonl_line():
    mod = _load_batch_module()
    req = mod.build_batch_request(
        custom_id="clip-1",
        body={"model": "gpt-4o-mini"},
    )
    line = mod.batch_request_to_jsonl_line(req)
    assert "\"custom_id\":\"clip-1\"" in line
    assert "\"url\":\"/v1/chat/completions\"" in line


def test_write_jsonl(tmp_path: Path):
    mod = _load_batch_module()
    reqs = [
        mod.build_batch_request("a", {"model": "gpt"}),
        mod.build_batch_request("b", {"model": "gpt"}),
    ]
    output = mod.write_jsonl(tmp_path / "batch.jsonl", reqs)
    data = output.read_text().strip().split("\n")
    assert len(data) == 2

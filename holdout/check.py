"""Step 0: list live Crusoe model ids and smoke-test the chosen models. `python -m holdout.check`"""
from __future__ import annotations

from . import config as C
from .llm import Totals, chat, client, extract_json

if __name__ == "__main__":
    ids = sorted(m.id for m in client().models.list().data)
    print("Live models:\n  " + "\n  ".join(ids))
    for role, model, temp in (("researcher", C.RESEARCHER_MODEL, C.RESEARCHER_TEMP), ("curator", C.CURATOR_MODEL, C.CURATOR_TEMP)):
        assert model in ids, f"{model} not live"
        r = chat(model, [{"role": "user", "content": 'Reply with JSON only: {"ok": true, "role": "%s"}' % role}],
                 temperature=temp, max_tokens=64, role=role, use_cache=False)
        print(f"{role:10s} {model}: {r.latency_s:.2f}s json={extract_json(r.text)} usage=({r.prompt_tokens} in, "
              f"{r.completion_tokens} out, cached={r.cached_tokens})")
    print("totals", Totals.as_dict())

"""Normalize the existing prompt files without loading any models."""
import json
from pathlib import Path


def load_records(path=None, text="", datasets=None, ids="", limit=0):
    if path:
        path = Path(path)
        if path.suffix.lower() == ".txt":
            records = [{"id": str(i), "prompt": p.strip()} for i, p in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if p.strip()]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("samples", data.get("scenarios", data.get("prompts")))
            if not isinstance(data, list):
                raise ValueError("JSON 应包含 samples、scenarios、prompts 数组，或直接是数组。")
            records = []
            for i, item in enumerate(data, 1):
                if isinstance(item, str):
                    record = {"id": str(i), "prompt": item}
                elif isinstance(item, dict):
                    record = {"id": str(item.get("source_sample_id", item.get("id", i))),
                              "dataset": str(item.get("dataset_key", "")),
                              "prompt": item.get("image_prompt", item.get("reference_image", item.get("prompt"))),
                              "negative_prompt": item.get("negative_prompt", "")}
                else:
                    raise ValueError(f"第 {i} 条记录格式错误。")
                if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
                    raise ValueError(f"第 {i} 条记录缺少提示词。")
                record["prompt"] = record["prompt"].strip()
                records.append(record)
    else:
        records = [{"id": str(i), "prompt": p.strip()} for i, p in enumerate(text.splitlines(), 1) if p.strip()]
    selected_ids = set(ids.replace(",", " ").split())
    if datasets:
        records = [r for r in records if r.get("dataset") in datasets]
    if selected_ids:
        records = [r for r in records if r["id"] in selected_ids]
        missing = selected_ids - {r["id"] for r in records}
        if missing:
            raise ValueError("未找到 ID：" + ", ".join(sorted(missing)))
    if int(limit) > 0:
        records = records[:int(limit)]
    if not records:
        raise ValueError("没有可生成的提示词，请检查输入与筛选条件。")
    return records

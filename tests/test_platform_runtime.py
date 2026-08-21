from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from hiob_core.platform import placement, runs


class FakeQuery:
    def __init__(self, client: "FakeClient", table: str):
        self.client = client
        self.table_name = table
        self.operation = "select"
        self.payload: Any = None
        self.filters: list[tuple[str, str, Any]] = []
        self.orders: list[tuple[str, bool]] = []
        self.limit_count: int | None = None

    def select(self, *_args: Any, **_kwargs: Any) -> "FakeQuery":
        return self

    def insert(self, payload: Any) -> "FakeQuery":
        self.operation = "insert"
        self.payload = deepcopy(payload)
        return self

    def update(self, payload: dict[str, Any]) -> "FakeQuery":
        self.operation = "update"
        self.payload = deepcopy(payload)
        return self

    def eq(self, key: str, value: Any) -> "FakeQuery":
        self.filters.append(("eq", key, value))
        return self

    def in_(self, key: str, values: list[Any]) -> "FakeQuery":
        self.filters.append(("in", key, values))
        return self

    def order(self, key: str, *, desc: bool = False) -> "FakeQuery":
        self.orders.append((key, desc))
        return self

    def limit(self, count: int) -> "FakeQuery":
        self.limit_count = count
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        for kind, key, value in self.filters:
            if kind == "eq" and row.get(key) != value:
                return False
            if kind == "in" and row.get(key) not in value:
                return False
        return True

    def execute(self) -> SimpleNamespace:
        self.client.calls.append(
            (self.table_name, self.operation, deepcopy(self.payload), list(self.filters))
        )
        if self.operation == "insert":
            if self.client.fail_insert_once == self.table_name:
                self.client.fail_insert_once = None
                raise RuntimeError("simulated old schema")
            items = self.payload if isinstance(self.payload, list) else [self.payload]
            inserted = []
            for item in items:
                row = deepcopy(item)
                row.setdefault("id", self.client.next_id(self.table_name))
                self.client.rows[self.table_name].append(row)
                inserted.append(deepcopy(row))
            return SimpleNamespace(data=inserted)

        matched = [row for row in self.client.rows[self.table_name] if self._matches(row)]
        if self.operation == "update":
            for row in matched:
                row.update(deepcopy(self.payload))
            return SimpleNamespace(data=deepcopy(matched))

        for key, desc in reversed(self.orders):
            matched.sort(key=lambda row: str(row.get(key) or ""), reverse=desc)
        if self.limit_count is not None:
            matched = matched[: self.limit_count]
        return SimpleNamespace(data=deepcopy(matched))


class FakeClient:
    def __init__(self, **rows: list[dict[str, Any]]):
        self.rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for table, values in rows.items():
            self.rows[table] = deepcopy(values)
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.counters: defaultdict[str, int] = defaultdict(int)
        self.fail_insert_once: str | None = None

    def next_id(self, table: str) -> str:
        self.counters[table] += 1
        return f"{table}-{self.counters[table]}"

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)


@pytest.fixture
def media_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def ensure_timeline(client: FakeClient, *, run_id: str, duration_ms: int, aspect: str):
        for row in client.rows["timeline"]:
            if row.get("run_id") == run_id:
                return row
        row = {
            "id": client.next_id("timeline"),
            "run_id": run_id,
            "duration_ms": duration_ms,
            "aspect": aspect,
        }
        client.rows["timeline"].append(row)
        return row

    def ensure_track(
        client: FakeClient,
        *,
        timeline_id: str,
        kind: str,
        label: str,
        ord: int,
        z_index: int,
    ):
        for row in client.rows["timeline_track"]:
            if row.get("timeline_id") == timeline_id and row.get("kind") == kind:
                return row
        row = {
            "id": client.next_id("timeline_track"),
            "timeline_id": timeline_id,
            "kind": kind,
            "label": label,
            "ord": ord,
            "z_index": z_index,
        }
        client.rows["timeline_track"].append(row)
        return row

    def create_clip(client: FakeClient, **payload: Any):
        row = {"id": client.next_id("clip"), **deepcopy(payload)}
        client.rows["clip"].append(row)
        return row

    monkeypatch.setattr(placement, "ensure_timeline", ensure_timeline)
    monkeypatch.setattr(placement, "ensure_track", ensure_track)
    monkeypatch.setattr(placement, "create_clip", create_clip)


@pytest.fixture
def library_item() -> dict[str, Any]:
    return {
        "id": "library-1",
        "kind": "video",
        "storage_key": "brand/demo.mp4",
        "sha256": "abc",
        "mime": "video/mp4",
        "duration_ms": 2500,
        "bytes": 10,
        "width": 1080,
        "height": 1920,
    }


def test_placement_private_helpers_and_pure_resolution(library_item: dict[str, Any]) -> None:
    client = FakeClient()
    assert placement._next_artifact_version(client, "slot-1") == 1
    client.rows["artifact"].append({"id": "old", "slot_id": "slot-1", "version": 2})
    assert placement._next_artifact_version(client, "slot-1") == 3

    created = placement._ensure_slot(
        client, run_id="run-1", track="visual", beat_index=0, start_ms=0, end_ms=1000
    )
    assert placement._ensure_slot(
        client, run_id="run-1", track="visual", beat_index=0, start_ms=9, end_ms=10
    ) == created
    artifact = placement._artifact_from_item(
        client, run_id="run-1", slot_id=created["id"], item=library_item, reuse="test"
    )
    assert artifact["attributes"]["reuse"] == "test"

    assert placement._resolve_placements([], {}, 1000, 0, at_ms=321) == [
        {"beat": 0, "start_ms": 321, "duration_ms": 1000}
    ]
    assert placement._resolve_placements([2], {}, 1000, 0, at_ms=321)[0]["beat"] == 2
    assert placement._resolve_placements([2, 3], {}, 1000, 500, mode="append") == [
        {"beat": 2, "start_ms": 500, "duration_ms": 1000},
        {"beat": 3, "start_ms": 1500, "duration_ms": 1000},
    ]
    assert placement._resolve_placements(
        [1, 4], {1: (100, 0)}, 1000, 800, mode="at_beat"
    ) == [
        {"beat": 1, "start_ms": 100, "duration_ms": 1000},
        {"beat": 4, "start_ms": 800, "duration_ms": 1000},
    ]


def test_place_media_validation_and_empty_resolution(
    monkeypatch: pytest.MonkeyPatch,
    media_helpers: None,
    library_item: dict[str, Any],
) -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="beats"):
        placement._place_media(
            client,
            run_id="r",
            item=library_item,
            beats=[],
            duration_ms=None,
            aspect="9:16",
            effects=None,
            reuse="x",
        )
    with pytest.raises(ValueError, match="storage_key"):
        placement._place_media(
            client,
            run_id="r",
            item={},
            beats=[0],
            duration_ms=None,
            aspect="9:16",
            effects=None,
            reuse="x",
        )
    monkeypatch.setattr(placement, "_resolve_placements", lambda *_a, **_k: [])
    with pytest.raises(ValueError, match="no placements"):
        placement._place_media(
            client,
            run_id="r",
            item=library_item,
            beats=[0],
            duration_ms=None,
            aspect="9:16",
            effects=None,
            reuse="x",
        )


def test_place_library_video_at_beats_and_exact_time(
    media_helpers: None, library_item: dict[str, Any]
) -> None:
    client = FakeClient(
        timeline=[{"id": "tl", "run_id": "r", "duration_ms": 8000, "aspect": "9:16"}],
        timeline_track=[{"id": "vt", "timeline_id": "tl", "kind": "video"}],
        clip=[{"id": "old", "track_id": "vt", "start_ms": 100, "duration_ms": 0, "beat_index": 0}],
    )
    result = placement.place_library_video(
        client, run_id="r", item=library_item, beats=[2, 0, 2], attributes={"source": "human"}
    )
    assert result["beats"] == [0, 2]
    assert len(result["clip_ids"]) == 2
    assert client.rows["timeline"][0]["duration_ms"] == 8000

    exact = placement.place_library_video(
        client, run_id="r", item={**library_item, "kind": None}, at_ms=9000, duration_ms=1000
    )
    assert exact["start_ms"] == 9000
    assert client.rows["timeline"][0]["duration_ms"] == 10000

    with pytest.raises(ValueError, match="expected video"):
        placement.place_library_video(
            client, run_id="r", item={**library_item, "kind": "image"}, beats=[0]
        )


def test_social_proof_treatments_and_quote(
    media_helpers: None, library_item: dict[str, Any]
) -> None:
    client = FakeClient()
    image_item = {**library_item, "kind": "image", "mime": "image/png", "storage_key": "review.png"}
    image_result = placement.place_social_proof(
        client, run_id="r", item=image_item, beat=1, quote="Verified", mode="append"
    )
    assert image_result["proof_frame"] is True
    assert image_result["quote_clip_id"]
    assert placement._is_image_item({"kind": "other", "mime": "image/jpeg"}) is True
    assert placement._is_image_item({"kind": "other", "mime": None}) is False

    video_result = placement.place_social_proof(client, run_id="r", item=library_item, beat=2)
    assert video_result["treatment"] == "full-bleed video"
    assert "quote_clip_id" not in video_result


@pytest.mark.parametrize("stage", ["timeline", "track", "clip"])
def test_demo_broll_missing_dependencies(
    stage: str, media_helpers: None, library_item: dict[str, Any]
) -> None:
    rows: dict[str, list[dict[str, Any]]] = {}
    if stage != "timeline":
        rows["timeline"] = [{"id": "tl", "run_id": "r"}]
    if stage == "clip":
        rows["timeline_track"] = [{"id": "vt", "timeline_id": "tl", "kind": "video"}]
    client = FakeClient(**rows)
    with pytest.raises(ValueError):
        placement.place_demo_broll(client, run_id="r", item=library_item, beat=2)


def test_demo_broll_success(media_helpers: None, library_item: dict[str, Any]) -> None:
    client = FakeClient(
        timeline=[{"id": "tl", "run_id": "r"}],
        timeline_track=[{"id": "vt", "timeline_id": "tl", "kind": "video"}],
        clip=[{"id": "c", "track_id": "vt", "beat_index": 2, "start_ms": 4000, "duration_ms": None}],
    )
    result = placement.place_demo_broll(client, run_id="r", item=library_item, beat=2)
    assert result["clip_id"] == "c"
    assert client.rows["clip"][0]["artifact_id"] == result["artifact_id"]


def test_sticker_validation_and_success(media_helpers: None, library_item: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="beats required"):
        placement.place_sticker(FakeClient(), run_id="r", item=library_item, beats=[])
    with pytest.raises(ValueError, match="no timeline"):
        placement.place_sticker(FakeClient(), run_id="r", item=library_item, beats=[1])
    with pytest.raises(ValueError, match="no video track"):
        placement.place_sticker(
            FakeClient(timeline=[{"id": "tl", "run_id": "r"}]),
            run_id="r",
            item=library_item,
            beats=[1],
        )
    base = {
        "timeline": [{"id": "tl", "run_id": "r"}],
        "timeline_track": [{"id": "vt", "timeline_id": "tl", "kind": "video"}],
    }
    with pytest.raises(ValueError, match="none of"):
        placement.place_sticker(
            FakeClient(**base, clip=[{"track_id": "vt", "beat_index": None, "start_ms": 0}]),
            run_id="r",
            item=library_item,
            beats=[9],
        )
    client = FakeClient(
        **base,
        clip=[
            {"track_id": "vt", "beat_index": None, "start_ms": 0, "duration_ms": 1},
            {"track_id": "vt", "beat_index": 1, "start_ms": 1000, "duration_ms": 500},
            {"track_id": "vt", "beat_index": 2, "start_ms": 1500, "duration_ms": None},
        ],
    )
    result = placement.place_sticker(
        client,
        run_id="r",
        item=library_item,
        beats=[2, 1, 2, 99],
        transforms={"scale": 0.5},
        effects=[{"kind": "pulse"}],
    )
    assert result["beats"] == [1, 2]
    assert len(result["clip_ids"]) == 2


def test_run_status_gate_and_basic_writes() -> None:
    client = FakeClient(run=[])
    assert runs.get_run_script_status(client, "r") is None
    client.rows["run"] = [{"id": "r", "script_status": None}]
    assert runs.get_run_script_status(client, "r") is None
    client.rows["run"][0]["script_status"] = "approved"
    assert runs.get_run_script_status(client, "r") == "approved"
    assert runs.assert_run_script_gate(client, "r", operation="produce") == "approved"
    with pytest.raises(RuntimeError, match="blocked"):
        runs.assert_run_script_gate(client, "missing", operation="render", allowed={"produced"})

    assert runs.set_run_script_status(client, "r", "queued")["script_status"] == "queued"
    assert runs.set_run_script_status(client, "missing", "queued") == {}
    assert runs.end_run(client, "r", output_url="x")["status"] == "succeeded"
    assert runs.end_run(client, "missing") == {}


def test_production_job_lifecycle_and_completion_gate() -> None:
    client = FakeClient(run=[{"id": "r", "script_status": "approved"}])
    assert runs.update_production_job(client, None, status="running") == {}
    first = runs.create_production_job(
        client,
        run_id="r",
        kind="visual",
        script_candidate_id="script-1",
        target={"beat": 0},
    )
    second = runs.create_production_job(client, run_id="r", kind="voiceover")
    assert first["script_candidate_id"] == "script-1"
    assert second["target"] == {}
    running = runs.update_production_job(
        client,
        first["id"],
        status="running",
        span_id="span",
        modal_call_id="call",
        error={},
        attributes={"a": 1},
    )
    assert running["started_at"] == "now()"
    terminal = runs.update_production_job(client, first["id"], status="cancelled")
    assert terminal["ended_at"] == "now()"
    queued = runs.update_production_job(client, first["id"], status="queued")
    assert queued["status"] == "queued"

    assert runs.maybe_mark_run_produced(FakeClient(), "r") == {}
    incomplete = FakeClient(
        production_jobs=[
            {"id": "a", "run_id": "r", "kind": "visual", "status": "succeeded", "queued_at": "2"},
            {"id": "b", "run_id": "r", "kind": "visual", "status": "failed", "queued_at": "1"},
            {"id": "c", "run_id": "r", "kind": "unknown", "status": "succeeded", "queued_at": "3"},
        ]
    )
    assert runs.maybe_mark_run_produced(incomplete, "r") == {}

    def jobs(statuses: dict[str, str]) -> FakeClient:
        return FakeClient(
            run=[{"id": "r", "script_status": "approved"}],
            production_jobs=[
                {"id": kind, "run_id": "r", "kind": kind, "status": status, "queued_at": "1", "created_at": "1"}
                for kind, status in statuses.items()
            ],
        )

    required_ok = {kind: "succeeded" for kind in runs.REQUIRED_PRODUCTION_WORK_KINDS}
    assert runs.maybe_mark_run_produced(jobs({**required_ok, "caption": "running"}), "r") == {}
    assert runs.maybe_mark_run_produced(jobs({**required_ok, "music": "failed"}), "r") == {}
    complete = jobs({**required_ok, "title_style": "skipped"})
    assert runs.maybe_mark_run_produced(complete, "r")["script_status"] == "produced"


def test_span_slot_and_text_artifact_writes() -> None:
    client = FakeClient()
    span = runs.start_span(
        client,
        run_id="r",
        name="n",
        kind="llm",
        service="core",
        attributes={"x": 1},
    )
    assert span["attributes"] == {"x": 1}
    ended = runs.end_span(
        client,
        span["id"],
        status="error",
        output_preview="bad",
        error={"message": "x"},
        attributes_patch={"tokens_in": 3},
    )
    assert ended["error"]["message"] == "x"
    assert ended["attributes"] == {"x": 1, "tokens_in": 3}
    assert runs.end_span(client, "missing", attributes_patch={"x": 1}) == {}
    assert runs.end_span(client, "missing") == {}

    slot = runs.create_slot(
        client, run_id="r", track="script", start_ms=0, end_ms=1000, beat_index=0
    )
    artifact = runs.create_text_artifact(
        client,
        run_id="r",
        slot_id=slot["id"],
        text="hello",
        role_code="writer",
        category="script",
        attributes={"locale": "en"},
    )
    assert artifact["role_code"] == "writer"
    assert client.rows["slot"][0]["current_artifact_id"] == artifact["id"]

    legacy_slot = runs.create_slot(client, run_id="r", track="script", start_ms=0, end_ms=1)
    client.fail_insert_once = "artifact"
    legacy = runs.create_text_artifact(
        client,
        run_id="r",
        slot_id=legacy_slot["id"],
        text="x",
        sha256="f" * 64,
    )
    assert "role_code" not in legacy
    assert legacy["storage_key"] == "inline:" + "f" * 16

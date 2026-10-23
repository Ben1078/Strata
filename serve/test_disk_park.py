"""serve/test_disk_park.py - disk parking of one conversation (DiskPark), offline: a simulated engine
(no GPU, no pack) that follows the engine's resume rules (live session / checkpoints, SAVE, RESTORE).

    python -m unittest serve.test_disk_park -v
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, DiskPark, Service, StrataEngine, _dp_lcp, _dp_prefix_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MARK = "Conversation-Id: main-chat"
GEN = [65, 66, 67]                                   # what the simulated model writes ("ABC")


class SimProc:
    """`strata --serve` as far as parking goes: GEN resumes from the live session when the prompt starts with all of
    it, else from the deepest checkpoint that starts it; a checkpoint is taken at the end of every prompt; SAVE
    writes the live session and the deepest checkpoint; RESTORE replaces both (the outgoing session is not parked)."""

    def __init__(self, engine):
        self.e = engine
        self.stdin = self
        self.sent: list[str] = []
        self.live: list[int] = []
        self.checks: list[list[int]] = []
        self.files: dict[str, tuple] = {}
        self.serr = None                             # next session command answers this SERR line
        self.park_written: list[int] = []
        self.bytes_per_token = 16000

    def write(self, s):
        line = s.rstrip("\n")
        cmd, _, rest = line.partition(" ")
        self.sent.append(cmd)
        q = self.e.lines
        if cmd in ("SAVE", "RESTORE", "PARKSAVE", "PARKLOAD") and self.serr:
            q.put(self.serr + "\n")
            self.serr = None
            return
        if cmd == "SAVE":
            deepest = max(self.checks, key=len) if self.checks else []
            self.files[rest] = (list(self.live), [list(deepest)] if deepest else [])
            nbytes = self.bytes_per_token * len(self.live)
            Path(rest).write_bytes(b"s" * 64)        # stands in for the file (its size is in the reply)
            os.truncate(rest, nbytes)
            q.put(f"SESSION {nbytes} {nbytes}\n")
            q.put(f"SAVED {len(self.live)} {nbytes} 12.5\n")
        elif cmd == "RESTORE":
            live, checks = self.files[rest]
            self.live, self.checks = list(live), [list(c) for c in checks]
            q.put(f"RESTORED {len(live)} {os.path.getsize(rest)} 20.0\n")
        elif cmd == "PARKSAVE":                      # park-file engine: the deepest checkpoint, appended when it extends
            deepest = max(self.checks, key=len) if self.checks else list(self.live)
            old = self.files.get(rest)
            extends = old is not None and deepest[:len(old[0])] == old[0]
            written = self.bytes_per_token * (len(deepest) - len(old[0]) if extends else len(deepest)) + 120_000_000
            self.files[rest] = (list(deepest), [])
            size = self.bytes_per_token * len(deepest)
            Path(rest).write_bytes(b"p" * 64)
            os.truncate(rest, size)
            self.park_written.append(written)
            q.put(f"SWAIT flush 70\n")
            q.put(f"PARKED {len(deepest)} {written} 9.5 {size} {'append' if extends else 'full'}\n")
        elif cmd == "PARKLOAD":
            live, _ = self.files[rest]
            self.live, self.checks = list(live), []
            q.put(f"SESSION 10 10\n")
            q.put(f"UNPARKED {len(live)} {self.bytes_per_token * len(live)} 15.0\n")
        elif cmd == "GEN":
            ids = [int(x) for x in rest.split(" ")[-1].split(",")]
            n = len(ids)
            resume, from_live = 0, False
            if self.live and len(self.live) <= n - 1 and ids[:len(self.live)] == self.live:
                resume, from_live = len(self.live), True
            for c in self.checks:
                if len(c) > resume and len(c) <= n - 1 and ids[:len(c)] == c:
                    resume, from_live = len(c), False
            if resume == 0:
                self.checks = []                     # another conversation (its own chain from here)
            self.checks = (self.checks + [list(ids)])[-8:]
            self.live = ids + GEN[:-1]
            q.put(f"RESUME {resume}\n")
            for t in GEN:
                q.put(f"T {t}\n")
            self.e.reads.append(n - resume)
            q.put(f"DONE {len(GEN)} {n} 1.0 1.0 length 0 0 {resume} 0 0 0 0 0 {n - resume} 0\n")
        elif cmd == "STOP":
            pass

    def flush(self):
        pass

    def poll(self):
        return None

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class SimEngine(StrataEngine):
    def __init__(self):
        super().__init__("strata", ["--max-context", "1000000", "--conversation-cache-min-free-mib", "2560"],
                         lazy=True)
        self.max_context = 1000000
        self.unloaded, self.ended = False, False
        self.lines = queue.Queue()
        self.can_stop = True
        self.reads: list[int] = []
        self.gen = 1
        self.proc = SimProc(self)


def sysmsg(extra="", mark=True):
    return {"role": "system", "content": "You are the main assistant.\nPlatform: chat-app\n" + (MARK if mark else "") +
            "\n" + extra}


TOOLS = [{"type": "function", "function": {"name": f"t{i}", "description": "x", "parameters": {"type": "object"}}}
         for i in range(3)]


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "main.session")
        self.engine = SimEngine()
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.mem = [64 << 30]
        self.svc.disk_park = self.park()

    def park(self, **over):
        cfg = {"file": self.path, "match": {"system_contains": ["Platform: chat-app", MARK], "min_tools": 2},
               "fork_markers": ["Review the conversation above"],
               "min_tokens": 2000, "tail_slack": 64, "min_gain": 500, "min_new_tokens": 0, "max_gb_per_day": 1000}
        cfg.update(over)
        return DiskPark(cfg, self.engine.spawn[1], mem_available=lambda: self.mem[0])

    def tearDown(self):
        self.dir.cleanup()

    def ask(self, messages, tools=TOOLS):
        ids, thinking, max_new = self.svc.prepare(messages, tools, {"enable_thinking": False}, 8)
        out = [x for k, x in self.svc.run(ids, thinking, tools, max_new, {}, threading.Event()) if k == "done"]
        return ids, out[-1]

    def cmds(self):
        c = [x for x in self.engine.proc.sent if x in ("SAVE", "RESTORE", "PARKSAVE", "PARKLOAD")]
        self.engine.proc.sent.clear()
        return c


def history(n_turns, size=1200, tag="h"):
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"{tag}{i} " + "lorem ipsum " * (size // 12)})
        msgs.append({"role": "assistant", "content": f"reply {i} ABC"})
    return msgs


class Classify(Base):
    def test_main_fork_side(self):
        dp = self.svc.disk_park
        self.assertEqual(dp.classify([sysmsg(), {"role": "user", "content": "hi"}], TOOLS), "main")
        self.assertEqual(dp.classify([sysmsg(mark=False), {"role": "user", "content": "hi"}], TOOLS), "side")
        self.assertEqual(dp.classify([sysmsg(), {"role": "user", "content": "hi"}], TOOLS[:1]), "side")   # min_tools
        self.assertEqual(dp.classify([{"role": "user", "content": "You are the main assistant " + MARK}], TOOLS), "side")
        review = [sysmsg(), {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                  {"role": "user", "content": "Review the conversation above and consider saving to memory."}]
        self.assertEqual(dp.classify(review, TOOLS), "fork")
        parts = [{"role": "system", "content": [{"type": "text", "text": sysmsg()["content"]}]},
                 {"role": "user", "content": "x"}]
        self.assertEqual(dp.classify(parts, TOOLS), "main")

    def test_prefix_hash_matches_js_stringify(self):
        body = {"messages": [{"role": "system", "content": "S é \"q\"\n"}, {"role": "user", "content": "u"}],
                "tools": [{"type": "function", "function": {"name": "a", "parameters": {"n": 3, "b": True}}}]}
        h = _dp_prefix_hash(body["messages"], body["tools"])
        dp = self.park(match={"prefix_hashes": [h]})
        self.assertEqual(dp.classify(body["messages"], body["tools"], raw=body), "main")
        self.assertEqual(dp.classify(body["messages"], body["tools"], raw={**body, "tools": []}), "side")

    def test_lcp(self):
        self.assertEqual(_dp_lcp([1, 2, 3], [1, 2, 3, 4]), 3)
        self.assertEqual(_dp_lcp([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(_dp_lcp([], [1]), 0)
        a = list(range(100000))
        b = a[:77777] + [-1] + a[77778:]
        self.assertEqual(_dp_lcp(a, b), 77777)


class Policy(Base):
    def test_side_parks_and_next_turn_restores(self):
        h = [sysmsg()] + history(4)
        ids1, _ = self.ask(h + [{"role": "user", "content": "turn A"}])
        self.assertEqual(self.cmds(), [])
        self.ask([{"role": "system", "content": "You are a side service."}, {"role": "user", "content": "derive"}])
        self.assertEqual(self.cmds(), ["SAVE"])                              # parked before the side request
        self.assertTrue(os.path.exists(self.path + ".json"))
        ids2, done = self.ask(h + [{"role": "user", "content": "turn A"}, {"role": "assistant", "content": "ABC"},
                                   {"role": "user", "content": "turn B"}])
        self.assertEqual(self.cmds(), ["RESTORE"])
        self.assertGreaterEqual(done["reused"], len(ids1) - 64)             # only the new tokens were read
        self.assertLess(self.engine.reads[-1], 200)
        # no-park reference: the same three requests without disk parking read the whole second turn
        ref = SimEngine()
        svc = Service(ref, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        for m in (h + [{"role": "user", "content": "turn A"}],
                  [{"role": "system", "content": "You are a side service."}, {"role": "user", "content": "derive"}]):
            i, t, mx = svc.prepare(m, TOOLS, {"enable_thinking": False}, 8)
            list(svc.run(i, t, TOOLS, mx, {}, threading.Event()))
        i, t, mx = svc.prepare(h + [{"role": "user", "content": "turn A"}, {"role": "assistant", "content": "ABC"},
                                    {"role": "user", "content": "turn B"}], TOOLS, {"enable_thinking": False}, 8)
        self.assertEqual(i, ids2)
        list(svc.run(i, t, TOOLS, mx, {}, threading.Event()))
        self.assertEqual(ref.reads[-1], len(ids2))                          # without parking: everything again

    def test_tool_loop_without_side_requests_touches_no_file(self):
        h = [sysmsg()] + history(4)
        self.ask(h + [{"role": "user", "content": "go"}])
        self.ask(h + [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ABC"},
                      {"role": "user", "content": "tool result"}])
        self.assertEqual(self.cmds(), [])
        self.assertLess(self.engine.reads[-1], 200)

    def test_unchanged_state_is_not_saved_twice(self):
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        self.ask(m1)
        side = [{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}]
        self.ask(side)
        self.assertEqual(self.cmds(), ["SAVE"])
        self.ask(side)                                                       # nothing new of the main chat
        self.assertEqual(self.cmds(), [])
        self.ask(m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}])
        self.ask(side)
        self.assertEqual(self.cmds(), ["RESTORE", "SAVE"])                   # the new turn is parked (overwrite)

    def test_fork_restores_but_never_overwrites(self):
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        ids1, _ = self.ask(m1)
        side = [{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}]
        self.ask(side)
        self.cmds()
        review = m1 + [{"role": "assistant", "content": "ABC"},
                       {"role": "user", "content": "Review the conversation above and save to memory."}]
        self.ask(review)
        self.assertEqual(self.cmds(), ["RESTORE"])
        self.assertLess(self.engine.reads[-1], 200)
        self.ask(side)
        self.assertEqual(self.cmds(), [])                                    # the fork's state is not parked
        self.assertEqual(len(self.svc.disk_park.file["ids"]), len(ids1))
        _, done = self.ask(m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}])
        self.assertEqual(self.cmds(), ["RESTORE"])
        self.assertGreaterEqual(done["reused"], len(ids1) - 64)

    def test_main_turn_right_after_fork_restores(self):
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        ids1, _ = self.ask(m1)
        self.ask(m1 + [{"role": "assistant", "content": "ABC"},
                       {"role": "user", "content": "Review the conversation above please."}])
        self.assertEqual(self.cmds(), ["SAVE"])                              # the fork waits for the park, not main
        _, done = self.ask(m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}])
        self.assertEqual(self.cmds(), ["RESTORE"])
        self.assertGreaterEqual(done["reused"], len(ids1) - 64)

    def test_compressed_chat_is_served_normally_then_overwrites(self):
        h = [sysmsg()] + history(4)
        self.ask(h + [{"role": "user", "content": "A"}])
        side = [{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}]
        self.ask(side)
        self.cmds()
        small = [sysmsg()] + [{"role": "user", "content": "[summary] " + "s " * 1500}]
        ids_c, done = self.ask(small)
        self.assertEqual(self.cmds(), [])                                    # no restore: the file does not match
        self.assertEqual(done["reused"], 0)
        self.ask(side)
        self.assertEqual(self.cmds(), ["SAVE"])                              # the file now holds the new chat
        self.assertEqual(len(self.svc.disk_park.file["ids"]), len(ids_c))

    def test_ram_floor_skips_the_restore(self):
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        ids1, _ = self.ask(m1)
        self.ask([{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}])
        self.cmds()
        need = self.svc.disk_park.restore_need() + (2560 << 20) + (512 << 20)
        self.mem[0] = need - 1
        _, done = self.ask(m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}])
        self.assertEqual(self.cmds(), [])
        self.assertEqual(self.svc.disk_park.stats["restore_skipped_ram"], 1)
        self.assertTrue(os.path.exists(self.path + ".json"))                 # kept for a later turn

    def test_refusals(self):
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        self.ask(m1)
        side = [{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}]
        self.ask(side)
        self.cmds()
        self.engine.proc.serr = "SERR memory 0 not enough RAM to read it"
        _, done = self.ask(m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}])
        self.assertEqual(done["finish"], "length")                           # served normally
        self.assertIsNotNone(self.svc.disk_park.file)                         # a RAM refusal keeps the file
        self.ask(side)
        self.cmds()
        self.engine.proc.serr = "SERR invalid 0 session file: saved with another model (model fingerprint differs)"
        self.ask(m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "C"}])
        self.assertIsNone(self.svc.disk_park.file)                            # an invalid file is dropped
        self.assertFalse(os.path.exists(self.path + ".json"))

    def test_engine_restart_forgets_the_live_state(self):
        h = [sysmsg()] + history(4)
        self.ask(h + [{"role": "user", "content": "A"}])
        self.engine.gen += 1                                                  # another engine process
        self.ask([{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}])
        self.assertEqual(self.cmds(), [])                                     # nothing to save: it is gone

    def test_file_survives_a_server_restart(self):
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        ids1, _ = self.ask(m1)
        self.ask([{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}])
        self.cmds()
        self.svc.disk_park = self.park()                                      # a new server reads the description
        self.assertEqual(len(self.svc.disk_park.file["ids"]), len(ids1))
        _, done = self.ask(m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}])
        self.assertEqual(self.cmds(), ["RESTORE"])
        self.assertGreaterEqual(done["reused"], len(ids1) - 64)

    def test_short_chats_are_left_to_the_ram_cache(self):
        m = [sysmsg(), {"role": "user", "content": "short"}]
        self.ask(m)
        self.ask([{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}])
        self.assertEqual(self.cmds(), [])


class Wear(Base):
    def test_small_growth_is_not_saved_again(self):
        self.svc.disk_park = self.park(min_new_tokens=5000)
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        ids1, _ = self.ask(m1)
        side = [{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}]
        self.ask(side)
        self.assertEqual(self.cmds(), ["SAVE"])                               # the first park: no file yet
        m2 = m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}]
        self.ask(m2)
        self.ask(side)
        self.assertEqual(self.cmds(), ["RESTORE"])                            # +few tokens: no new 16 KB/token write
        self.assertEqual(self.svc.disk_park.stats["skipped_small"], 1)
        m3 = m2 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "C"}]
        _, done = self.ask(m3)
        self.assertEqual(self.cmds(), ["RESTORE"])                            # the older file still serves
        self.assertGreaterEqual(done["reused"], len(ids1) - 64)
        self.ask(side)
        self.cmds()
        big = m3 + [{"role": "assistant", "content": "ABC"}] + history(6, tag="g")[:-1]
        self.ask(big)
        self.ask(side)
        self.assertEqual(self.cmds(), ["RESTORE", "SAVE"])                    # enough new tokens: parked again

    def test_daily_budget(self):
        h = [sysmsg()] + history(4)
        side = [{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}]
        m1 = h + [{"role": "user", "content": "A"}]
        self.ask(m1)
        self.ask(side)
        self.assertEqual(self.cmds(), ["SAVE"])
        self.svc.disk_park.max_bytes_per_day = self.svc.disk_park.wear["bytes"] + (100 << 20)   # < one more save
        m2 = m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}]
        self.ask(m2)
        self.ask(side)
        self.assertEqual(self.cmds(), ["RESTORE"])                            # over the budget: not written
        self.assertEqual(self.svc.disk_park.stats["skipped_budget"], 1)
        w = json.loads(Path(self.path + ".wear.json").read_text())
        self.assertEqual(w["saves"], 1)


class EnginePark(Base):
    """The engine's park files (PARKSAVE / PARKLOAD)."""
    def setUp(self):
        super().setUp()
        self.svc.disk_park = self.park(engine_park=True, min_new_tokens=0)

    def test_flow_appends_and_streams(self):
        h = [sysmsg()] + history(4)
        side = [{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}]
        m1 = h + [{"role": "user", "content": "A"}]
        ids1, _ = self.ask(m1)
        self.ask(side)
        self.assertEqual(self.cmds(), ["PARKSAVE"])
        f = self.svc.disk_park.file
        self.assertEqual(len(f["ids"]), len(ids1))           # the deepest checkpoint = this prompt
        m2 = m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}]
        _, done = self.ask(m2)
        self.assertEqual(self.cmds(), ["PARKLOAD"])
        self.assertGreaterEqual(done["reused"], len(ids1) - 64)
        self.assertLess(self.engine.reads[-1], 200)
        self.ask(side)
        self.assertEqual(self.cmds(), ["PARKSAVE"])
        w = self.engine.proc.park_written
        self.assertLess(w[1] - 120_000_000, (w[0] - 120_000_000) / 3)   # the second park appended
        self.assertEqual(self.svc.disk_park.wear["bytes"], sum(w))   # wear counts what was written
        _, done = self.ask(m2 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "C"}])
        self.assertEqual(self.cmds(), ["PARKLOAD"])
        self.assertLess(self.engine.reads[-1], 200)
        self.assertEqual(self.svc.disk_park.stats["last"]["restore"]["tokens"], len(self.svc.disk_park.file["ids"]))

    def test_streamed_load_needs_little_ram(self):
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        self.ask(m1)
        self.ask([{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}])
        self.cmds()
        self.svc.disk_park.file["bytes"] = 9 << 30         # a 9 GB file: a whole-file read would not fit
        self.mem[0] = (2560 << 20) + (1 << 30)             # 1 GiB above the floor
        self.ask(m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}])
        self.assertEqual(self.cmds(), ["PARKLOAD"])

    def test_opt_out_request(self):
        h = [sysmsg()] + history(4)
        m1 = h + [{"role": "user", "content": "A"}]
        self.ask(m1)
        self.ask([{"role": "system", "content": "Side service"}, {"role": "user", "content": "x"}])
        self.cmds()
        m2 = m1 + [{"role": "assistant", "content": "ABC"}, {"role": "user", "content": "B"}]
        ids, thinking, max_new = self.svc.prepare(m2, TOOLS, {"enable_thinking": False}, 8, req={"strata_disk_park": False})
        list(self.svc.run(ids, thinking, TOOLS, max_new, {}, threading.Event()))
        self.assertEqual(self.cmds(), [])                      # neither loaded nor parked
        self.assertIsNone(self.svc.disk_park.live)
        self.ask([{"role": "system", "content": "Side service"}, {"role": "user", "content": "y"}])
        self.assertEqual(self.cmds(), [])                      # nothing to park afterwards either

    def test_defaults(self):
        dp = DiskPark({"file": self.path, "engine_park": True}, [])
        self.assertEqual((dp.min_new_tokens, dp.max_bytes_per_day), (4096, 50 * 10 ** 9))


if __name__ == "__main__":
    unittest.main()

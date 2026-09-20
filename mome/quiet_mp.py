#!/usr/bin/env python3
"""Drop multiprocessing child-bootstrap noise from standard error.

Why this exists.  The symbolic equivalence check puts a timeout on every
comparison.  On Unix that is `signal.alarm`; Windows has no such call, so the
library spawns a child process per comparison instead (its own documentation
warns this "will incur a huge performance penalty").  Across tens of thousands
of spawns a child occasionally dies during bootstrap and prints a traceback:

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
        from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=...)
      ...
    EOFError: Ran out of input          (or OSError: [WinError 6] ...)

None of this affects the parent's computation: that one comparison is treated
as a timeout and the run continues.  The problem is the log.  Buried under
this noise, a real error goes unseen.

What this does NOT do: it does not change any verdict.  The timeout, the
comparison and the result are untouched; only that block of standard error is
discarded.  A registered number must not move because of a logging fix.

    from quiet_mp import drop_spawn_bootstrap_noise
    with drop_spawn_bootstrap_noise():
        ...
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading

#: Markers that appear only in a child-bootstrap traceback.  A line carrying
#: one belongs to such a block: this package never uses multiprocessing
#: directly, so there is no other source for them.
_MARKS = (
    "multiprocessing.spawn",
    "spawn_main",
    'File "<string>", line 1, in <module>',
    "multiprocessing\\spawn.py",
    "multiprocessing/spawn.py",
    "multiprocessing\\reduction.py",
    "multiprocessing/reduction.py",
    "_winapi.DuplicateHandle",
    "reduction.duplicate",
    "reduction.pickle.load(from_parent)",
    "parent_sentinel",
)
#: The exception line such a block ends on.  When several children write at
#: once the lines interleave and this one can arrive on its own, which is why
#: it is matched separately.
_NOISE_EXC = ("OSError: [WinError 6]", "EOFError: Ran out of input")
#: The shape of an exception line.  A line that ends the DROP state is
#: discarded when it matches and emitted when it does not, so that a real
#: error interleaved into the noise still reaches the log.
_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit)\b")


def _is_noise(line: str) -> bool:
    return any(m in line for m in _MARKS) or line.startswith(_NOISE_EXC)


class BootstrapNoiseFilter:
    """A line-by-line state machine, pure enough to test without processes.

    `feed(line)` returns the lines to emit, which is empty for a line to drop.
    """

    NORMAL, MAYBE, DROP = 0, 1, 2

    def __init__(self) -> None:
        self.state = self.NORMAL
        self.pending: list[str] = []
        self.dropped = 0

    def feed(self, line: str) -> list[str]:
        if self.state == self.NORMAL:
            if line.startswith("Traceback (most recent call last)"):
                self.state = self.MAYBE
                self.pending = [line]
                return []
            #: With several children the output interleaves and a fragment can
            #: arrive with no block header.  A line carrying a marker belongs
            #: to such a block regardless, so dropping starts there.
            if _is_noise(line):
                self.dropped += 1
                self.state = self.DROP
                return []
            return [line]

        if self.state == self.MAYBE:
            #: Several children can send header lines back to back.  Passing
            #: them through leaves bare `Traceback ...` lines piling up on the
            #: screen, so the headers collapse to one and the machine keeps
            #: waiting: a marker after them drops the lot, a real frame emits
            #: one header along with it.  Frames and exception lines are never
            #: lost either way, so a real error stays readable.
            if line.startswith("Traceback (most recent call last)"):
                return []
            #: A bootstrap marker on the second line condemns the whole
            #: block.  Anything else releases the held lines untouched, which
            #: is how a real traceback survives.
            if any(m in line for m in _MARKS):
                self.state = self.DROP
                self.pending = []
                self.dropped += 1
                return []
            out = self.pending + [line]
            self.pending = []
            self.state = self.NORMAL
            return out

        # DROP: indented lines (continuing frames) and further block headers
        # are dropped, and the state holds.
        if line[:1].isspace() or line.startswith("Traceback (most recent call last)"):
            return []
        # An unindented line ends the block, on that line.  The state has to
        # be restored first: leaving it in DROP swallows a real traceback that
        # follows immediately.  The line itself is this block's last if it is
        # an exception line or a marker, and is emitted otherwise.
        self.state = self.NORMAL
        return [] if (_EXC_LINE.match(line) or _is_noise(line)) else [line]

    def flush(self) -> list[str]:
        out, self.pending = self.pending, []
        self.state = self.NORMAL
        return out


def _pump(read_fd: int, out_fd: int, filt: BootstrapNoiseFilter) -> None:
    with os.fdopen(read_fd, "rb", 0) as r:
        buf = b""
        while True:
            chunk = r.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                for line in filt.feed(raw.decode("utf-8", "replace")):
                    os.write(out_fd, (line + "\n").encode("utf-8", "replace"))
        if buf:
            for line in filt.feed(buf.decode("utf-8", "replace")):
                os.write(out_fd, (line + "\n").encode("utf-8", "replace"))
        for line in filt.flush():
            os.write(out_fd, (line + "\n").encode("utf-8", "replace"))


@contextlib.contextmanager
def drop_spawn_bootstrap_noise(report: bool = True):
    """Remove child-bootstrap tracebacks from standard error inside the block.

    The swap happens at the file-descriptor level so that child output is
    caught too; replacing `sys.stderr` in Python only would let children write
    straight past it.  If the plumbing cannot be set up the block does nothing
    and passes everything through, because losing a real error while chasing
    noise is the worse failure.
    """
    try:
        saved = os.dup(2)
        read_fd, write_fd = os.pipe()
    except OSError:
        yield
        return
    filt = BootstrapNoiseFilter()
    thread = threading.Thread(target=_pump, args=(read_fd, saved, filt), daemon=True)
    thread.start()
    try:
        os.dup2(write_fd, 2)
        os.close(write_fd)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved, 2)
        thread.join(timeout=5)
        if report and filt.dropped:
            print(f"(dropped {filt.dropped} multiprocessing child-bootstrap "
                  f"tracebacks from stderr; results are unaffected. "
                  f"See quiet_mp.py)",
                  file=sys.stderr, flush=True)
        os.close(saved)

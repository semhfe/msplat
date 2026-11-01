#!/usr/bin/env python3
"""Checks that the helpers copied into core/metal/msplat_q.metal are the text of core/metal/msplat_metal.metal.

Pre-registration s0 (Metal prerequisite): the projection and SH helpers used by the Q kernels are text copies inside
namespace mq, compiled with the default numeric settings of the original. In msplat_q.metal:
  - every block after a "// @copy" marker line (up to the next marker or "// ---- end of copies ----") must occur
    byte-identically, as one contiguous piece, in msplat_metal.metal;
  - every "// @derived <name>" block has its own signature, and its body (from the first "{") must be byte-identical
    to the body of the source function <name>;
  - every #define line before "namespace mq {" must be a line of msplat_metal.metal.
Exit 0 = all copies identical. Usage: q_copy_check.py [--selftest] [repo_root]
"""
import re
import sys
from pathlib import Path


def source_body(src, name):
    m = re.search(r'^(?:inline )?\w+ ' + re.escape(name) + r'\(', src, re.M)
    if not m:
        return None
    i = src.index('{', m.end())
    depth = 0
    for k in range(i, len(src)):
        depth += {'{': 1, '}': -1}.get(src[k], 0)
        if depth == 0:
            return src[i:k + 1]
    return None


def check(q, src):
    errors, n = [], 0
    head = q[:q.index('namespace mq {')]
    src_lines = set(src.split('\n'))
    for line in head.split('\n'):
        if line.startswith('#define'):
            n += 1
            if line not in src_lines:
                errors.append('define not in source: ' + line)
    region = q[q.index('namespace mq {'):q.index('// ---- end of copies ----')]
    parts = re.split(r'^(// @(?:copy|derived)[^\n]*)\n', region, flags=re.M)
    for marker, block in zip(parts[1::2], parts[2::2]):
        block = block.rstrip('\n')
        n += 1
        if marker.startswith('// @copy'):
            if block not in src:
                errors.append('copy differs from source: ' + block.split('\n')[0])
        else:
            name = marker.split()[2]
            want = source_body(src, name)
            if want is None or block[block.index('{'):] != want:
                errors.append('derived body differs from ' + name)
    return n, errors


def main():
    args = [a for a in sys.argv[1:] if a != '--selftest']
    root = Path(args[0]) if args else Path(__file__).resolve().parent.parent
    q = (root / 'core/metal/msplat_q.metal').read_text()
    src = (root / 'core/metal/msplat_metal.metal').read_text()
    if '--selftest' in sys.argv:
        bad = q.replace('1e-8f', '1e-7f', 1)          # one changed character inside quat_to_rotmat
        assert bad != q and check(bad, src)[1], 'selftest: mutated copy not detected'
        bad = q.replace('param -= step_size', 'param -=  step_size', 1)   # inside the derived Adam body
        assert bad != q and check(bad, src)[1], 'selftest: mutated derived body not detected'
        print('selftest ok')
    n, errors = check(q, src)
    for e in errors:
        print('FAIL', e)
    print('%d blocks checked, %d differ -> %s' % (n, len(errors), 'FAIL' if errors or n == 0 else 'PASS'))
    return 1 if errors or n == 0 else 0


if __name__ == '__main__':
    sys.exit(main())

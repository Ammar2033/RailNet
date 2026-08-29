import numpy as np

from ._compile import compile_exact_routes_exhaustive

SAFE_SCAN_MAX_MISSING = 48
SAFE_SCAN_LIMIT = 160
RESIDUAL_CANDIDATES = 64
REPAIR_COMPILE_BUDGET = 96


def repair_missing_values(target_values, target_bits, counts, rails, max_terms, verbose=False):
    """
    Targeted post-learning repair.

    Analysis showed remaining missing values are extreme tail
    values skipped by rank-uniform init. Each missing value can
    often become 1-term exact by placing its own bits on an
    unused / least-used rail.

    Strategy:
        Phase 1 (batch): place top missing values into ALL
        strictly-unused rails at once, single compile check.
        Phase 2 (single): bounded swap trials on least-used rails.

    Accept only if exhaustive exact count improves.
    Monotone best-so-far. Hard-capped compile count for CPU time.
    """

    rails = rails.copy()

    total = len(target_bits)

    def compile_count(r):
        table = compile_exact_routes_exhaustive(target_bits, r, max_terms)
        c = 0
        for b in target_bits:
            if int(b) in table:
                c += 1
        return c, table

    compiles = 0

    best_count, _ = compile_count(rails)
    compiles += 1

    if best_count == total:
        return rails

    # --------------------------------------------------------
    # Rail usage histogram from current routes.
    # --------------------------------------------------------

    def usage_histogram(table):
        usage = np.zeros(len(rails), dtype=np.int64)
        for route in table.values():
            for rid, _sign in route:
                usage[rid] += 1
        return usage

    # --------------------------------------------------------
    # PHASE 1: batch placement into unused rails.
    # --------------------------------------------------------

    if compiles < REPAIR_COMPILE_BUDGET:
        _, table_now = compile_count(rails)
        compiles += 1

        usage = usage_histogram(table_now)

        zero_slots = [i for i in range(len(rails)) if usage[i] == 0]

        existing = {int(x) for x in rails}

        missing_items = []

        for i in range(total):
            b = int(target_bits[i])

            if b not in table_now:
                missing_items.append((int(counts[i]), b))

        missing_items.sort(reverse=True)

        if zero_slots and missing_items:
            trial = rails.copy()

            placed = []

            for _cnt, cand_bits in missing_items:
                if not zero_slots:
                    break

                if cand_bits in existing:
                    continue

                slot = zero_slots.pop(0)

                trial[slot] = np.uint16(cand_bits)

                existing.add(cand_bits)

                placed.append(slot)

            if placed:
                trial_count, _trial_table = compile_count(trial)

                compiles += 1

                if trial_count > best_count:
                    best_count = trial_count

                    rails = trial

                    if verbose:
                        print(
                            f"  [repair-batch] placed "
                            f"{len(placed)} missing -> exact "
                            f"{best_count}/{total}",
                            flush=True,
                        )

        if best_count == total:
            return rails

    # --------------------------------------------------------
    # PHASE 2: single swap trials (bounded).
    # --------------------------------------------------------

    while compiles < REPAIR_COMPILE_BUDGET:
        _, table_now = compile_count(rails)
        compiles += 1

        current_count = sum(1 for b in target_bits if int(b) in table_now)

        if current_count >= total:
            break

        usage = usage_histogram(table_now)

        slot_order = np.argsort(usage)[:8]

        existing = {int(x) for x in rails}

        missing_items = []

        for i in range(total):
            b = int(target_bits[i])

            if b not in table_now:
                missing_items.append((int(counts[i]), b))

        missing_items.sort(reverse=True)

        missing_items = missing_items[:12]

        improved = False

        for cand_slot in slot_order:
            if compiles >= REPAIR_COMPILE_BUDGET:
                break

            for _cnt, cand_bits in missing_items:
                if cand_bits in existing:
                    continue

                saved = int(rails[cand_slot])

                rails[cand_slot] = np.uint16(cand_bits)

                trial_count, _ttable = compile_count(rails)

                compiles += 1

                if trial_count > current_count:
                    current_count = trial_count

                    if current_count > best_count:
                        best_count = trial_count

                    existing.add(cand_bits)
                    existing.discard(saved)

                    improved = True

                    if verbose:
                        print(
                            f"  [repair] slot {cand_slot}: "
                            f"{saved:04X} -> {cand_bits:04X} "
                            f"exact {current_count}/{total}",
                            flush=True,
                        )

                    break

                rails[cand_slot] = np.uint16(saved)

            if improved and compiles < REPAIR_COMPILE_BUDGET:
                continue

        if not improved:
            break

    return rails


# ============================================================
# SAFE SLOT REPAIR (final gap closer)
# ============================================================


def repair_safe_slots(target_values, target_bits, counts, rails, max_terms, verbose=False):
    """
    Final-gap closer for the last few missing values.

    A rail slot is SAFE if removing it does not reduce the
    exact coverage (its own bit is still representable via
    other rails). Safe slots are free real estate: placing a
    missing value there is a guaranteed pure gain.

    Cost: one exhaustive compile per scanned rail. Bounded by
    SAFE_SCAN_LIMIT and only run when the missing count is
    small.
    """

    total = len(target_bits)

    original = rails.copy()

    def compile_count(r):
        table = compile_exact_routes_exhaustive(target_bits, r, max_terms)
        c = 0
        for b in target_bits:
            if int(b) in table:
                c += 1
        return c, table

    base_count, base_table = compile_count(rails)

    if base_count == total:
        return rails

    missing_items = []

    for i in range(total):
        b = int(target_bits[i])

        if b not in base_table:
            missing_items.append((int(counts[i]), b))

    if not missing_items:
        return rails

    # Only worth scanning when close to full coverage.
    if len(missing_items) > SAFE_SCAN_MAX_MISSING:
        return rails

    missing_items.sort(reverse=True)

    existing = {int(x) for x in rails}

    # Sentinel far outside tensor range; combos with it can
    # never equal any target, so its slot is effectively empty.
    sentinel = np.uint16(0x4280)

    usage = np.zeros(len(rails), dtype=np.int64)

    for route in base_table.values():
        for rid, _sign in route:
            usage[rid] += 1

    scan_order = np.argsort(usage)

    safe_slots = []

    scanned = 0

    for slot in scan_order:
        if len(safe_slots) >= len(missing_items):
            break

        if scanned >= SAFE_SCAN_LIMIT:
            break

        saved_bits = int(rails[slot])

        if saved_bits == int(sentinel):
            continue

        rails[slot] = sentinel

        trial_count, _trial_table = compile_count(rails)

        scanned += 1

        if trial_count >= base_count:
            safe_slots.append(slot)

        rails[slot] = np.uint16(saved_bits)

    if not safe_slots:
        return rails

    # --------------------------------------------------------
    # Batch-place missing values into safe slots.
    # --------------------------------------------------------

    placed = 0

    for _cnt, cand_bits in missing_items:
        if not safe_slots:
            break

        if cand_bits in existing:
            continue

        slot = safe_slots.pop(0)

        rails[slot] = np.uint16(cand_bits)

        existing.add(cand_bits)

        placed += 1

    if placed == 0:
        return rails

    final_count, _ftable = compile_count(rails)

    if final_count < base_count:
        # Should not happen (safe slots are pure gains),
        # but keep monotone guarantee anyway.
        return original

    if verbose and placed:
        print(
            f"  [safe-repair] placed {placed} missing into safe slots -> {final_count}/{total}",
            flush=True,
        )

    return rails


# ============================================================
# BASIS COORDINATE UPDATE
# ============================================================

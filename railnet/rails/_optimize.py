import numpy as np

from railnet.dtypes.bf16 import (
    bf16_array_to_float32,
    bf16_bits_to_float32,
    float32_to_bf16_bits,
)

from ._compile import (
    calculate_objective,
    compile_exact_routes_exhaustive,
    greedy_routes,
    reconstruct_routes,
)
from ._init import initialize_rails
from ._repair import repair_missing_values, repair_safe_slots

MAX_RAIL_REPAIRS_PER_ITER = 4
RESIDUAL_CANDIDATES = 64
REPAIR_COMPILE_BUDGET = 96


def update_basis(target_values, counts, rails, routes, signs, exact_mask_values):
    """
    Coordinate-descent update.

    For each rail, estimate the best shared rail value
    from all weights currently routed through that rail.

    The resulting value is quantized back to BF16.

    Important:
        - No per-weight coefficient is introduced.
        - Rails remain BF16 primitives.
        - `exact_mask_values` is kept in the signature for
          compatibility with the optimizer pipeline.
    """

    rail_count = len(rails)

    # Current BF16 rail values as float64 for optimization.
    rail_values = bf16_array_to_float32(rails).astype(np.float64)

    max_terms = routes.shape[1]

    for rail_id in range(rail_count):
        numerator = 0.0
        denominator = 0.0

        # ----------------------------------------------------
        # Find all route positions that use this rail.
        # ----------------------------------------------------

        for term in range(max_terms):
            ids = routes[:, term]

            active = ids == rail_id + 1

            if not np.any(active):
                continue

            active_indices = np.flatnonzero(active)

            signs_active = signs[active_indices, term].astype(np.float64)

            # ------------------------------------------------
            # Reconstruct the contribution of all OTHER rails
            # for these targets.
            # ------------------------------------------------

            current = np.zeros(len(active_indices), dtype=np.float64)

            for other_term in range(max_terms):
                if other_term == term:
                    continue

                other_ids = routes[active_indices, other_term]

                other_signs = signs[active_indices, other_term].astype(np.float64)

                other_active = other_ids > 0

                # Exclude the current rail explicitly.
                other_active &= other_ids != rail_id + 1

                if not np.any(other_active):
                    continue

                other_zero_based = other_ids[other_active] - 1

                current[other_active] += other_signs[other_active] * rail_values[other_zero_based]

            # ------------------------------------------------
            # Desired total contribution of this rail:
            #
            #   target = current + sign * rail
            #
            # therefore:
            #
            #   rail = sign * (target - current)
            # ------------------------------------------------

            desired = target_values[active_indices] - current

            signed_desired = signs_active * desired

            weights = counts[active_indices]

            numerator += float(np.sum(weights * signed_desired))

            denominator += float(np.sum(weights))

        # ----------------------------------------------------
        # No weight currently uses this rail.
        # ----------------------------------------------------

        if denominator <= 0.0:
            continue

        # ----------------------------------------------------
        # Weighted optimal value for this rail.
        # ----------------------------------------------------

        new_value = numerator / denominator

        # ----------------------------------------------------
        # Hardware primitive remains BF16.
        # ----------------------------------------------------

        new_bits = float32_to_bf16_bits(new_value)

        # ----------------------------------------------------
        # Update both representations.
        # ----------------------------------------------------

        rail_values[rail_id] = bf16_bits_to_float32(new_bits).astype(np.float64)

        rails[rail_id] = np.uint16(new_bits)

    return rails


# ============================================================
# DUPLICATE RAIL REPAIR
# ============================================================


def score_objective(objective):
    """
    Spec section 29 ranking:

        1. exact_unique (full coverage first)
        2. weighted_exact
        3. -weighted_mse
    """

    return (
        int(objective["exact_unique"]),
        float(objective["weighted_exact"]),
        -float(objective["weighted_mse"]),
    )


# ============================================================
# LEARN ONE BASIS
# ============================================================


def repair_duplicate_rails(rails, target_values, counts, routes, signs, residual):
    """
    If two learned rails collapse to the same BF16 value,
    replace the least useful duplicate rail with a high-frequency
    residual candidate.
    """

    seen = {}

    duplicates = []

    for i, value in enumerate(rails):
        key = int(value)

        if key in seen:
            duplicates.append(i)
        else:
            seen[key] = i

    if not duplicates:
        return rails

    # Residual candidates:
    # values not currently well represented.
    score = counts * np.abs(residual)

    order = np.argsort(score)[::-1]

    used = {int(x) for x in rails}

    replacement_index = 0

    for rail_index in duplicates:
        while replacement_index < len(order):
            candidate_value = float(target_values[order[replacement_index]])

            replacement_index += 1

            candidate_bits = float32_to_bf16_bits(candidate_value)

            if candidate_bits in used:
                continue

            rails[rail_index] = np.uint16(candidate_bits)

            used.add(candidate_bits)

            break

    return rails


# ============================================================
# LOCAL RAIL REPAIR
# ============================================================


def try_residual_repairs(target_values, target_bits, counts, rails, max_terms, current_best_score):
    """
    Try a small number of high-value residual candidates.

    This is NOT brute force.

    Only a few rails are replaced and accepted if the objective
    improves.
    """

    (current_routes, current_signs, residual, _) = greedy_routes(target_values, rails, max_terms)

    current_reconstructed = reconstruct_routes(current_routes, current_signs, rails)

    current_objective = calculate_objective(
        target_values, target_bits, counts, current_reconstructed
    )

    candidate_score = (
        current_objective["exact_unique"],
        current_objective["weighted_exact"],
        -current_objective["weighted_mse"],
    )

    current_best_score = max(current_best_score, candidate_score)

    residual_score = counts * np.abs(residual)

    order = np.argsort(residual_score)[::-1]

    candidate_indices = order[: min(RESIDUAL_CANDIDATES, len(order))]

    existing = {int(x) for x in rails}

    # Least useful rails are those with low route frequency.
    # Filter zero entries (unused term slots) to avoid bincount
    # ValueError on negative values after sparse routing.
    active_mask = current_routes > 0

    if np.any(active_mask):
        rail_usage = np.bincount((current_routes[active_mask] - 1), minlength=len(rails))
    else:
        rail_usage = np.zeros(len(rails), dtype=np.int64)

    least_useful = np.argsort(rail_usage)[
        # Cap scanned rails: full scans on failing bases are
        # extremely slow (each candidate costs a greedy pass).
        :10
    ]

    accepted = 0

    best_rails = rails.copy()
    best_score = candidate_score

    for rail_index in least_useful:
        if accepted >= MAX_RAIL_REPAIRS_PER_ITER:
            break

        for candidate_index in candidate_indices:
            candidate_bits = int(target_bits[candidate_index])

            if candidate_bits in existing:
                continue

            trial = rails.copy()

            trial[rail_index] = np.uint16(candidate_bits)

            (trial_routes, trial_signs, _, _) = greedy_routes(target_values, trial, max_terms)

            trial_reconstructed = reconstruct_routes(trial_routes, trial_signs, trial)

            trial_objective = calculate_objective(
                target_values, target_bits, counts, trial_reconstructed
            )

            trial_score = (
                trial_objective["exact_unique"],
                trial_objective["weighted_exact"],
                -trial_objective["weighted_mse"],
            )

            if trial_score > best_score:
                best_score = trial_score
                best_rails = trial
                accepted += 1
                # Keep existing in sync with best_rails
                existing = {int(x) for x in best_rails}
                break

    return (best_rails, best_score)


# ============================================================
# SCORE
# ============================================================


def learn_basis(
    target_values, target_bits, counts, rail_count, max_terms, verbose=False, max_iters=300
):

    rails = initialize_rails(target_values, target_bits, counts, rail_count)

    best_rails = rails.copy()
    best_routes = None
    best_signs = None
    best_objective = None
    best_score = (-1.0, -1, -float("inf"))

    previous_mse = float("inf")

    history = []
    CONVERGENCE_TOL = 0

    MAX_ITERS = int(max_iters)
    for iteration in range(1, MAX_ITERS + 1):
        if verbose:
            print(
                f"    [learn r={rail_count} t={max_terms}] iter {iteration}/{MAX_ITERS}", flush=True
            )

        # ----------------------------------------------------
        # Route
        # ----------------------------------------------------

        (routes, signs, residual, active_terms) = greedy_routes(target_values, rails, max_terms)

        # ----------------------------------------------------
        # Reconstruct
        # ----------------------------------------------------

        reconstructed = reconstruct_routes(routes, signs, rails)

        objective = calculate_objective(target_values, target_bits, counts, reconstructed)

        score = score_objective(objective)

        history.append(
            {
                "iteration": int(iteration),
                "exact_unique": int(objective["exact_unique"]),
                "weighted_exact": float(objective["weighted_exact"]),
                "weighted_mse": float(objective["weighted_mse"]),
                "max_error": float(objective["max_error"]),
                "active_terms": int(np.sum(active_terms)),
            }
        )

        # ----------------------------------------------------
        # Best-so-far
        # ----------------------------------------------------

        if score > best_score:
            best_score = score

            best_rails = rails.copy()

            best_routes = routes.copy()
            best_signs = signs.copy()

            best_objective = objective

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        current_mse = objective["weighted_mse"]

        improvement = previous_mse - current_mse

        previous_mse = current_mse

        if iteration > 1 and abs(improvement) < CONVERGENCE_TOL:
            # Still allow basis repair once.
            pass

        # ----------------------------------------------------
        # Update basis
        # ----------------------------------------------------

        updated_rails = update_basis(
            target_values, counts, rails.copy(), routes, signs, objective["exact_mask"]
        )

        updated_rails = repair_duplicate_rails(
            updated_rails, target_values, counts, routes, signs, residual
        )

        # ----------------------------------------------------
        # Evaluate new rails.
        # ----------------------------------------------------

        (next_routes, next_signs, _next_residual, _next_active) = greedy_routes(
            target_values, updated_rails, max_terms
        )

        next_reconstructed = reconstruct_routes(next_routes, next_signs, updated_rails)

        next_objective = calculate_objective(target_values, target_bits, counts, next_reconstructed)

        next_score = score_objective(next_objective)

        # ----------------------------------------------------
        # Accept update only if it improves.
        # This gives monotonic best-so-far behavior.
        # ----------------------------------------------------

        if next_score >= score:
            rails = updated_rails
        # else: keep current basis (no improvement).

        # ----------------------------------------------------
        # Small residual repair every second iteration.
        # ----------------------------------------------------

        if iteration % 2 == 0:
            repaired_rails, repaired_score = try_residual_repairs(
                target_values, target_bits, counts, rails, max_terms, best_score
            )

            if repaired_score > score:
                rails = repaired_rails

        # ----------------------------------------------------
        # Full exact stop.
        # ----------------------------------------------------

        if objective["exact_unique"] == len(target_values):
            break

    # --------------------------------------------------------
    # Final targeted repair of missing values using
    # exhaustive evaluation (monotone best-so-far).
    # --------------------------------------------------------

    if best_rails is not None:
        repaired_rails = repair_missing_values(
            target_values, target_bits, counts, best_rails, max_terms
        )

        repaired_rails = repair_safe_slots(
            target_values, target_bits, counts, repaired_rails, max_terms
        )

        repaired_table = compile_exact_routes_exhaustive(target_bits, repaired_rails, max_terms)

        repaired_exact = sum(1 for b in target_bits if int(b) in repaired_table)

        current_best_exact = best_objective["exact_unique"] if best_objective is not None else -1

        if repaired_exact >= current_best_exact:
            best_rails = repaired_rails

            # Refresh greedy route/objective on final rails so
            # returned representation stays consistent.
            (final_routes, final_signs, _res, _act) = greedy_routes(
                target_values, best_rails, max_terms
            )

            final_reconstructed = reconstruct_routes(final_routes, final_signs, best_rails)

            best_objective = calculate_objective(
                target_values, target_bits, counts, final_reconstructed
            )

            best_routes = final_routes
            best_signs = final_signs

    return {
        "rails": best_rails,
        "routes": best_routes,
        "signs": best_signs,
        "objective": best_objective,
        "score": best_score,
        "history": history,
    }


# ============================================================
# REPRESENTATION
# ============================================================

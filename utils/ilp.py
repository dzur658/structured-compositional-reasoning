from __future__ import annotations

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _is_neither_op(op: str) -> bool:
    return "NEITHER" in op.strip().upper()


def _build_tf(atomics: list[str], constraints: list[str], sc: dict) -> tuple[dict, dict]:
    """Pull normalized t/f (0-1) per (atomic, constraint) out of the scorer output."""
    t: dict = {}
    f: dict = {}
    for a in atomics:
        t[a] = {}
        f[a] = {}
        for c in constraints:
            entry = sc.get(a, {}).get(c, {})
            t[a][c] = (entry.get("T", 0) or 0) / 5
            f[a][c] = (entry.get("F", 0) or 0) / 5
    return t, f


def solve_hard_gate(
    options: dict,
    atomics: list[str],
    constraints: list[str],
    t: dict,
    f: dict,
) -> dict:
    """Each atomic is either fully valid or fully invalid. An AND/OR/NEITHER
    option is only eligible if the validity pattern of its two atomics
    actually satisfies the operator. Solved with scipy.optimize.milp."""
    A_list = list(atomics)
    K_list = list(constraints)
    nA, nK = len(A_list), len(K_list)
    a_idx = {a: i for i, a in enumerate(A_list)}
    c_idx = {c: i for i, c in enumerate(K_list)}

    opt_keys = [k for k in ["A", "B", "C", "D"] if k in options] or list(options.keys())
    nO = len(opt_keys)
    o_idx = {o: i for i, o in enumerate(opt_keys)}

    def z_i(a: int, k: int) -> int: return a * nK + k
    n_z = nA * nK
    off_y = n_z
    def y_i(a: int) -> int: return off_y + a
    off_x = off_y + nA
    def x_i(o: int) -> int: return off_x + o
    off_u = off_x + nO
    def u_i(o: int, r: int, k: int) -> int: return off_u + (o * 2 + r) * nK + k
    off_w = off_u + nO * 2 * nK
    def w_i(o: int, r: int, k: int) -> int: return off_w + (o * 2 + r) * nK + k
    n_vars = off_w + nO * 2 * nK

    if n_vars == 0 or nK == 0:
        return {"prediction": opt_keys[0] if opt_keys else None, "objective": 0.0,
                "status": "no_constraints", "x": {}, "y": {}, "z": {}}

    c_obj = np.zeros(n_vars)
    for ok, oi in o_idx.items():
        opt = options[ok]
        pair = [opt.get("a1"), opt.get("a2")]
        for r, a in enumerate(pair):
            if a not in a_idx:
                continue
            ai = a_idx[a]
            for cname, ki in c_idx.items():
                t_val = t.get(a, {}).get(cname, 0.0)
                f_val = f.get(a, {}).get(cname, 0.0)
                c_obj[u_i(oi, r, ki)] -= t_val
                c_obj[w_i(oi, r, ki)] -= f_val

    A_rows: list[np.ndarray] = []
    lb_list: list[float] = []
    ub_list: list[float] = []

    def add_le(coeffs: dict[int, float], rhs: float) -> None:
        row = np.zeros(n_vars)
        for idx, val in coeffs.items():
            row[idx] += val
        A_rows.append(row); lb_list.append(-np.inf); ub_list.append(rhs)

    def add_ge(coeffs: dict[int, float], rhs: float) -> None:
        row = np.zeros(n_vars)
        for idx, val in coeffs.items():
            row[idx] += val
        A_rows.append(row); lb_list.append(rhs); ub_list.append(np.inf)

    def add_eq(coeffs: dict[int, float], rhs: float) -> None:
        row = np.zeros(n_vars)
        for idx, val in coeffs.items():
            row[idx] += val
        A_rows.append(row); lb_list.append(rhs); ub_list.append(rhs)

    # y_a <= z[a,k]
    for ai in range(nA):
        for ki in range(nK):
            add_le({y_i(ai): 1, z_i(ai, ki): -1}, 0)

    # y_a >= sum_k z[a,k] - (K-1)
    for ai in range(nA):
        coeffs = {z_i(ai, ki): 1 for ki in range(nK)}
        coeffs[y_i(ai)] = coeffs.get(y_i(ai), 0) - 1
        add_le(coeffs, nK - 1)

    # operator eligibility
    for ok, oi in o_idx.items():
        opt = options[ok]
        op = opt["operator"].strip().upper()
        a1, a2 = opt.get("a1"), opt.get("a2")
        a1i, a2i = a_idx.get(a1), a_idx.get(a2)
        if op == "AND":
            if a1i is not None:
                add_le({x_i(oi): 1, y_i(a1i): -1}, 0)
            if a2i is not None:
                add_le({x_i(oi): 1, y_i(a2i): -1}, 0)
        elif op == "OR":
            coeffs = {x_i(oi): 1}
            if a1i is not None:
                coeffs[y_i(a1i)] = coeffs.get(y_i(a1i), 0) - 1
            if a2i is not None:
                coeffs[y_i(a2i)] = coeffs.get(y_i(a2i), 0) - 1
            add_le(coeffs, 0)
        elif _is_neither_op(op):
            if a1i is not None:
                add_le({x_i(oi): 1, y_i(a1i): 1}, 1)
            if a2i is not None:
                add_le({x_i(oi): 1, y_i(a2i): 1}, 1)

    # exactly one option selected
    add_eq({x_i(oi): 1 for oi in range(nO)}, 1)

    # u/w linearization (d substituted as 1 - z throughout)
    for ok, oi in o_idx.items():
        opt = options[ok]
        pair = [opt.get("a1"), opt.get("a2")]
        for r, a in enumerate(pair):
            ai = a_idx.get(a)
            for ki in range(nK):
                if ai is None:
                    add_eq({u_i(oi, r, ki): 1}, 0)
                    add_eq({w_i(oi, r, ki): 1}, 0)
                    continue
                add_le({u_i(oi, r, ki): 1, x_i(oi): -1}, 0)
                add_le({u_i(oi, r, ki): 1, z_i(ai, ki): -1}, 0)
                add_ge({u_i(oi, r, ki): 1, x_i(oi): -1, z_i(ai, ki): -1}, -1)
                add_le({w_i(oi, r, ki): 1, x_i(oi): -1}, 0)
                add_le({w_i(oi, r, ki): 1, z_i(ai, ki): 1}, 1)
                add_ge({w_i(oi, r, ki): 1, x_i(oi): -1, z_i(ai, ki): 1}, 0)

    A_mat = np.array(A_rows)
    lin_constraint = LinearConstraint(A_mat, lb=np.array(lb_list), ub=np.array(ub_list))
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)

    res = milp(c_obj, constraints=[lin_constraint], integrality=integrality, bounds=bounds)

    if res.x is None:
        return {"prediction": opt_keys[0], "objective": None, "status": "infeasible",
                "x": {}, "y": {}, "z": {}}

    xv = np.round(res.x)
    x_sel = {ok: int(xv[x_i(oi)]) for ok, oi in o_idx.items()}
    pred = next((ok for ok, v in x_sel.items() if v == 1), opt_keys[0])
    y_sel = {a: int(xv[y_i(a_idx[a])]) for a in A_list}
    z_sel = {a: {cn: int(xv[z_i(a_idx[a], ci)]) for cn, ci in c_idx.items()} for a in A_list}

    return {
        "prediction": pred,
        "objective": float(-res.fun) if res.fun is not None else None,
        "status": "optimal",
        "x": x_sel, "y": y_sel, "z": z_sel,
    }

#gurobi code
# def solve_lcsqa_gurobi_hard_gate(
#     options: dict,
#     atomics: list[str],
#     constraints: list[str],
#     t: dict,
#     f: dict,
# ) -> dict:
#     if not _GUROBI_AVAILABLE:
#         raise RuntimeError("gurobipy is not installed. Run: pip install gurobipy "
#                             "(and have a valid license) to use this backend, or "
#                             "use solve_lcsqa_milp_hard_gate (scipy) instead.")

#     opt_keys = [k for k in ["A", "B", "C", "D"] if k in options] or list(options.keys())
#     K = len(constraints)
#     a_idx = {a: i for i, a in enumerate(atomics)}
#     o_idx = {o: i for i, o in enumerate(opt_keys)}

#     m = gp.Model()
#     m.setParam("OutputFlag", 0)
#     m.setParam("TimeLimit", 30)

#     z = m.addVars(len(atomics), K, vtype=GRB.BINARY, name="z")
#     y = m.addVars(len(atomics), vtype=GRB.BINARY, name="y")
#     x = m.addVars(len(opt_keys), vtype=GRB.BINARY, name="x")
#     u = m.addVars(len(opt_keys), 2, K, vtype=GRB.BINARY, name="u")
#     w = m.addVars(len(opt_keys), 2, K, vtype=GRB.BINARY, name="w")

#     for ai in range(len(atomics)):
#         for ki in range(K):
#             m.addConstr(y[ai] <= z[ai, ki])              # (2a)
#             # m.addConstr(z[ai, ki] <= y[ai])               # (2c) -- the fix
#         if K > 0:
#             m.addConstr(y[ai] >= gp.quicksum(z[ai, ki] for ki in range(K)) - (K - 1))  # (2b)

#     for o, oi in o_idx.items():
#         opt = options[o]
#         op = opt["operator"].strip().upper()
#         a1i, a2i = a_idx.get(opt.get("a1")), a_idx.get(opt.get("a2"))
#         if op == "AND":
#             if a1i is not None: m.addConstr(x[oi] <= y[a1i])
#             if a2i is not None: m.addConstr(x[oi] <= y[a2i])
#         elif op == "OR":
#             rhs = gp.LinExpr()
#             if a1i is not None: rhs += y[a1i]
#             if a2i is not None: rhs += y[a2i]
#             m.addConstr(x[oi] <= rhs)
#         elif _is_neither_op(op):
#             if a1i is not None: m.addConstr(x[oi] <= 1 - y[a1i])
#             if a2i is not None: m.addConstr(x[oi] <= 1 - y[a2i])

#     m.addConstr(gp.quicksum(x[oi] for oi in range(len(opt_keys))) == 1)

#     for o, oi in o_idx.items():
#         opt = options[o]
#         pair = [opt.get("a1"), opt.get("a2")]
#         for r, a in enumerate(pair):
#             ai = a_idx.get(a)
#             for ki in range(K):
#                 if ai is None:
#                     m.addConstr(u[oi, r, ki] == 0)
#                     m.addConstr(w[oi, r, ki] == 0)
#                     continue
#                 m.addConstr(u[oi, r, ki] <= x[oi])
#                 m.addConstr(u[oi, r, ki] <= z[ai, ki])
#                 m.addConstr(u[oi, r, ki] >= x[oi] + z[ai, ki] - 1)
#                 m.addConstr(w[oi, r, ki] <= x[oi])
#                 m.addConstr(w[oi, r, ki] <= 1 - z[ai, ki])
#                 m.addConstr(w[oi, r, ki] >= x[oi] - z[ai, ki])

#     obj = gp.LinExpr()
#     for o, oi in o_idx.items():
#         opt = options[o]
#         pair = [opt.get("a1"), opt.get("a2")]
#         for r, a in enumerate(pair):
#             for ki, c in enumerate(constraints):
#                 t_val = t.get(a, {}).get(c, 0.0)
#                 f_val = f.get(a, {}).get(c, 0.0)
#                 obj += t_val * u[oi, r, ki] + f_val * w[oi, r, ki]
#     m.setObjective(obj, GRB.MAXIMIZE)
#     m.optimize()

#     if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
#         return {"prediction": opt_keys[0], "objective": None, "status": "infeasible",
#                 "x": {}, "y": {}, "z": {}}

#     x_sel = {o: int(round(x[oi].X)) for o, oi in o_idx.items()}
#     pred = next((o for o, v in x_sel.items() if v == 1), opt_keys[0])
#     y_sel = {a: int(round(y[a_idx[a]].X)) for a in atomics}
#     z_sel = {a: {c: int(round(z[a_idx[a], ki].X)) for ki, c in enumerate(constraints)} for a in atomics}

#     return {
#         "prediction": pred,
#         "objective": round(m.ObjVal, 6),
#         "status": "optimal",
#         "x": x_sel, "y": y_sel, "z": z_sel,
#     }



def _option_score_given_y(
    opt: dict, y_sel: dict, t: dict, f: dict, constraints: list[str],
) -> tuple[float, bool]:
    op = opt["operator"].strip().upper()
    a1, a2 = opt.get("a1"), opt.get("a2")
    y1, y2 = y_sel.get(a1, 0), y_sel.get(a2, 0)

    if op == "AND":
        eligible = (y1 == 1 and y2 == 1)
    elif op == "OR":
        eligible = (y1 == 1 or y2 == 1)
    elif _is_neither_op(op):
        eligible = (y1 == 0 and y2 == 0)
    else:
        eligible = False

    if not eligible:
        return -1.0, False

    score = 0.0
    for a in (a1, a2):
        y = y_sel.get(a, 0)
        for c in constraints:
            score += t.get(a, {}).get(c, 0.0) if y == 1 else f.get(a, {}).get(c, 0.0)
    return score, True


def select_answer(
    options: dict,
    atomics: list[str],
    constraints: list[str],
    sc: dict,
) -> dict:
    t, f = _build_tf(atomics, constraints, sc)
    hard = solve_hard_gate(options, atomics, constraints, t, f)

    if hard["status"] == "optimal":
        scores, eligible = {}, {}
        for ok, opt in options.items():
            score, elig = _option_score_given_y(opt, hard["y"], t, f, constraints)
            scores[ok] = score
            eligible[ok] = elig
    else:
        scores, eligible = {}, {}

    return {
        "prediction": hard["prediction"],
        "option_scores": scores,
        "eligible": eligible,
        "status": hard["status"],
        "y": hard.get("y", {}),
        "z": hard.get("z", {}),
    }

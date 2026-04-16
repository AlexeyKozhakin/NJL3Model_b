const NJL = (function () {
    'use strict';

    // -------------------------------------------------------------------------
    // Вспомогательные функции
    // -------------------------------------------------------------------------
    function midpoints(a, b, N) {
        const res = new Float64Array(N);
        const step = (b - a) / N;
        const start = a + step * 0.5;
        for (let i = 0; i < N; i++) {
            res[i] = start + i * step;
        }
        return res;
    }

    function linspace(a, b, N) {
        const res = new Float64Array(N);
        if (N === 1) {
            res[0] = a;
            return res;
        }
        const step = (b - a) / (N - 1);
        for (let i = 0; i < N; i++) {
            res[i] = a + i * step;
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // Omega_L (конечный размер L)
    // -------------------------------------------------------------------------
    function omegaL_int(u1, u3, L, b, M, phi) {
        const p1 = u1 / (1 - u1);
        const p3 = u3 / (1 - u3);
        const E1 = Math.sqrt(M * M + p1 * p1);
        const B_plus = Math.sqrt(p3 * p3 + (E1 + b) * (E1 + b));
        const B_minus = Math.sqrt(p3 * p3 + (E1 - b) * (E1 - b));
        const term1 = 1 - 2 * Math.cos(2 * Math.PI * phi) * Math.exp(-L * B_plus) + Math.exp(-2 * L * B_plus);
        const term2 = 1 - 2 * Math.cos(2 * Math.PI * phi) * Math.exp(-L * B_minus) + Math.exp(-2 * L * B_minus);
        return -4 * Math.log(term1 * term2) / (4 * Math.PI * Math.PI) / ((1 - u1) * (1 - u1)) / ((1 - u3) * (1 - u3)) / L;
    }

    function omegaL_scalar(L, b, M, N_h_p1, N_h_p2) {
        const p1_vals = midpoints(0, 1, N_h_p1);
        const p2_vals = midpoints(0, 1, N_h_p2);
        const dp1 = 1.0 / N_h_p1;
        const dp2 = 1.0 / N_h_p2;
        const phi = 0;
        let sum = 0.0;
        for (let i = 0; i < N_h_p1; i++) {
            const u1 = p1_vals[i];
            for (let j = 0; j < N_h_p2; j++) {
                const u3 = p2_vals[j];
                sum += omegaL_int(u1, u3, L, b, M, phi);
            }
        }
        return sum * dp1 * dp2;
    }

    // -------------------------------------------------------------------------
    // dU (контрчлен Уайтинга)
    // -------------------------------------------------------------------------
    function dU_int(u, phi, b, M) {
        const p = u / (1 - u);
        const term1 = M * M / p;
        const c = Math.cos(phi);
        const term2 = Math.sqrt(b * b + p * p + 2 * b * p * c);
        const term3 = Math.sqrt(b * b + p * p - 2 * b * p * c);
        const inner = Math.sqrt(M * M + p * p * c * c);
        const term4 = Math.sqrt(M * M + b * b + p * p + 2 * b * inner);
        const term5 = Math.sqrt(M * M + b * b + p * p - 2 * b * inner);
        return -(-term1 - term2 - term3 + term4 + term5) * p / (Math.PI * Math.PI) / ((1 - u) * (1 - u));
    }

    function dU_scalar(b, M, N_h_p, N_h_phi) {
        const p_vals = midpoints(0, 1, N_h_p);
        const phi_vals = midpoints(0, Math.PI / 2, N_h_phi);
        const dp = 1.0 / N_h_p;
        const dphi = (Math.PI / 2) / N_h_phi;
        let sum = 0.0;
        for (let i = 0; i < N_h_p; i++) {
            const u = p_vals[i];
            for (let j = 0; j < N_h_phi; j++) {
                const phi = phi_vals[j];
                sum += dU_int(u, phi, b, M);
            }
        }
        return sum * dp * dphi;
    }

    // -------------------------------------------------------------------------
    // Omega_mu_L (конечная плотность)
    // -------------------------------------------------------------------------
    function Fpn_plus(p1, n, M, b, L, phi, mu) {
        const E1 = Math.sqrt(M * M + p1 * p1);
        return (mu - Math.sqrt((E1 + b) * (E1 + b) + (2 * Math.PI / L * (n + phi)) ** 2)) / (2 * Math.PI);
    }

    function Fpn_minus(p1, n, M, b, L, phi, mu) {
        const E1 = Math.sqrt(M * M + p1 * p1);
        return (mu - Math.sqrt((E1 - b) * (E1 - b) + (2 * Math.PI / L * (n + phi)) ** 2)) / (2 * Math.PI);
    }

    function n_max_plus(M, b, L, mu, phi) {
        if (Math.abs(b) > M && b < 0) {
            const val = Fpn_plus(Math.sqrt(b * b - M * M), 0, M, b, L, phi, mu);
            if (val <= 0) return -1;
            return Math.floor(L * mu / (2 * Math.PI) - phi);
        } else if (mu * mu <= (M + b) * (M + b)) {
            return -1;
        } else {
            return Math.floor(L / (2 * Math.PI) * Math.sqrt(mu * mu - (M + b) * (M + b)) - phi);
        }
    }

    function n_max_minus(M, b, L, mu, phi) {
        if (Math.abs(b) > M && b > 0) {
            const val = Fpn_minus(Math.sqrt(b * b - M * M), 0, M, b, L, phi, mu);
            if (val <= 0) return -1;
            return Math.floor(L * mu / (2 * Math.PI) - phi);
        } else if (mu * mu <= (M - b) * (M - b)) {
            return -1;
        } else {
            return Math.floor(L / (2 * Math.PI) * Math.sqrt(mu * mu - (M - b) * (M - b)) - phi);
        }
    }

    function integrate_mode_scalar(funcCode, n, M, b, L, phi, mu, N_h, p_left, p_right) {
        if (p_right <= p_left) return 0.0;
        const dp = (p_right - p_left) / N_h;
        let total = 0.0;
        for (let i = 0; i < N_h; i++) {
            const p = p_left + (i + 0.5) * dp;
            const val = funcCode === 0
                ? Fpn_plus(p, n, M, b, L, phi, mu)
                : Fpn_minus(p, n, M, b, L, phi, mu);
            if (val > 0) total += val;
        }
        return total * dp;
    }

    function compute_Unp(M, b, L, mu, phi, N_h) {
        const Nmax = n_max_plus(M, b, L, mu, phi);
        if (Nmax < 0) return 0.0;
        let total = 0.0;
        for (let n = 0; n <= Nmax; n++) {
            const Sn = Math.sqrt(mu * mu - (2.0 * Math.PI * n / L) ** 2);
            const F0 = Fpn_plus(0.0, n, M, b, L, phi, mu);
            let two_roots = false;
            let p_left = 0.0, p_right = 0.0;
            if (Math.abs(b) > M && b < 0 && F0 < 0.0) {
                const val_left = (Math.abs(b) - Sn) ** 2 - M * M;
                const val_right = (Math.abs(b) + Sn) ** 2 - M * M;
                if (val_left >= 0.0 && val_right >= 0.0) {
                    two_roots = true;
                    p_left = Math.sqrt(val_left);
                    p_right = Math.sqrt(val_right);
                }
            }
            if (!two_roots) {
                let pr_sq;
                if (b >= 0) {
                    pr_sq = (Sn - b) ** 2 - M * M;
                } else {
                    pr_sq = (Sn + Math.abs(b)) ** 2 - M * M;
                }
                if (pr_sq < 0.0) continue;
                p_left = 0.0;
                p_right = Math.sqrt(pr_sq);
            }
            const integral = integrate_mode_scalar(0, n, M, b, L, phi, mu, N_h, p_left, p_right);
            total += (n === 0 ? 1.0 : 2.0) * integral;
        }
        return total;
    }

    function compute_Unm(M, b, L, mu, phi, N_h) {
        const Nmax = n_max_minus(M, b, L, mu, phi);
        if (Nmax < 0) return 0.0;
        let total = 0.0;
        for (let n = 0; n <= Nmax; n++) {
            const Sn = Math.sqrt(mu * mu - (2.0 * Math.PI * n / L) ** 2);
            const F0 = Fpn_minus(0.0, n, M, b, L, phi, mu);
            let two_roots = false;
            let p_left = 0.0, p_right = 0.0;
            if (Math.abs(b) > M && b > 0 && F0 < 0.0) {
                const val_left = (b - Sn) ** 2 - M * M;
                const val_right = (b + Sn) ** 2 - M * M;
                if (val_left >= 0.0 && val_right >= 0.0) {
                    two_roots = true;
                    p_left = Math.sqrt(val_left);
                    p_right = Math.sqrt(val_right);
                }
            }
            if (!two_roots) {
                let pr_sq;
                if (b >= 0) {
                    pr_sq = (Sn + b) ** 2 - M * M;
                } else {
                    pr_sq = (Sn - Math.abs(b)) ** 2 - M * M;
                }
                if (pr_sq < 0.0) continue;
                p_left = 0.0;
                p_right = Math.sqrt(pr_sq);
            }
            const integral = integrate_mode_scalar(1, n, M, b, L, phi, mu, N_h, p_left, p_right);
            total += (n === 0 ? 1.0 : 2.0) * integral;
        }
        return total;
    }

    function omegaMuL_scalar(mu, L, b, M, phi, N_h) {
        const Unp = compute_Unp(M, b, L, mu, phi, N_h);
        const Unm = compute_Unm(M, b, L, mu, phi, N_h);
        return -(2.0 / L) * (Unp + Unm);
    }

    // -------------------------------------------------------------------------
    // Основной расчёт сетки
    // -------------------------------------------------------------------------
    function computeGrid(params, onProgress) {
        const { mu, L, g, bMin, bMax, MMin, MMax, N_b, N_M,
                N_h_p1, N_h_p2, N_h_p, N_h_phi, N_h_mu } = params;

        const b_vals = linspace(bMin, bMax, N_b);
        const M_vals = linspace(MMin, MMax, N_M);

        // Предвычисление универсальных констант
        const omegaL_00 = omegaL_scalar(L, 0.0, 0.0, N_h_p1, N_h_p2);
        const omegaMuL_00 = omegaMuL_scalar(mu, L, 0.0, 0.0, 0, N_h_mu);

        const total   = new Float64Array(N_b * N_M);
        const omegaMuL= new Float64Array(N_b * N_M);
        const dU      = new Float64Array(N_b * N_M);
        const omegaL  = new Float64Array(N_b * N_M);

        for (let i = 0; i < N_b; i++) {
            const b = b_vals[i];
            const omegaL_b0 = omegaL_scalar(L, b, 0.0, N_h_p1, N_h_p2);
            const omegaMuL_b0 = omegaMuL_scalar(mu, L, b, 0.0, 0, N_h_mu);

            for (let j = 0; j < N_M; j++) {
                const M = M_vals[j];
                const idx = i * N_M + j;

                const omegaL_full = omegaL_scalar(L, b, M, N_h_p1, N_h_p2);
                const omegaL_phys = omegaL_full - omegaL_b0 + omegaL_00;
                omegaL[idx] = omegaL_phys;

                const dU_val = dU_scalar(b, M, N_h_p, N_h_phi);
                dU[idx] = dU_val;

                const omegaMuL_val = omegaMuL_scalar(mu, L, b, M, 0, N_h_mu) - omegaMuL_b0 + omegaMuL_00;
                omegaMuL[idx] = omegaMuL_val;

                total[idx] = (M * M) / (2.0 * g) + dU_val + omegaL_phys + omegaMuL_val;
            }
            if (onProgress) onProgress(i + 1, N_b);
        }

        // Поиск минимума
        let minVal = Infinity;
        let minI = 0, minJ = 0;
        for (let i = 0; i < N_b; i++) {
            for (let j = 0; j < N_M; j++) {
                const idx = i * N_M + j;
                if (total[idx] < minVal) {
                    minVal = total[idx];
                    minI = i;
                    minJ = j;
                }
            }
        }

        return {
            b_vals, M_vals,
            total, omegaMuL, dU, omegaL,
            b_min: b_vals[minI],
            M_min: M_vals[minJ],
            Omega_min: minVal
        };
    }

    // -------------------------------------------------------------------------
    // 1D-срезы по уже посчитанной сетке
    // -------------------------------------------------------------------------
    function extractSlice(grid, fixed_b, fixed_M) {
        const N_b = grid.b_vals.length;
        const N_M = grid.M_vals.length;

        // Срез при фиксированном b (варьируем M)
        let bIdx = 0;
        let minBDiff = Infinity;
        for (let i = 0; i < N_b; i++) {
            const diff = Math.abs(grid.b_vals[i] - fixed_b);
            if (diff < minBDiff) {
                minBDiff = diff;
                bIdx = i;
            }
        }

        // Срез при фиксированном M (варьируем b)
        let mIdx = 0;
        let minMDiff = Infinity;
        for (let j = 0; j < N_M; j++) {
            const diff = Math.abs(grid.M_vals[j] - fixed_M);
            if (diff < minMDiff) {
                minMDiff = diff;
                mIdx = j;
            }
        }

        const slice_b = new Float64Array(N_M);
        const slice_M = new Float64Array(N_b);
        const slice_b_omegaMuL = new Float64Array(N_M);
        const slice_b_dU = new Float64Array(N_M);
        const slice_b_omegaL = new Float64Array(N_M);
        const slice_M_omegaMuL = new Float64Array(N_b);
        const slice_M_dU = new Float64Array(N_b);
        const slice_M_omegaL = new Float64Array(N_b);

        for (let j = 0; j < N_M; j++) {
            const idx = bIdx * N_M + j;
            slice_b[j] = grid.total[idx];
            slice_b_omegaMuL[j] = grid.omegaMuL[idx];
            slice_b_dU[j] = grid.dU[idx];
            slice_b_omegaL[j] = grid.omegaL[idx];
        }
        for (let i = 0; i < N_b; i++) {
            const idx = i * N_M + mIdx;
            slice_M[i] = grid.total[idx];
            slice_M_omegaMuL[i] = grid.omegaMuL[idx];
            slice_M_dU[i] = grid.dU[idx];
            slice_M_omegaL[i] = grid.omegaL[idx];
        }

        return {
            b_idx: bIdx, b_actual: grid.b_vals[bIdx],
            M_idx: mIdx, M_actual: grid.M_vals[mIdx],
            slice_b, slice_M,
            slice_b_omegaMuL, slice_b_dU, slice_b_omegaL,
            slice_M_omegaMuL, slice_M_dU, slice_M_omegaL
        };
    }

    return {
        computeGrid,
        extractSlice,
        linspace,
        midpoints
    };
})();

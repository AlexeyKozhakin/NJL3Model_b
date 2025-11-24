
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <ctime>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Функция аналогичная Fpn_plus_numpy
inline double Fpn_plus(double p1, double n, double M, double b, double L, double phi, double mu) {
    double E1 = std::sqrt(M*M + p1*p1);
    double val = (mu - std::sqrt((E1 + b)*(E1 + b) + std::pow(2 * M_PI / L * (n + phi), 2))) / (2 * M_PI);
    return val > 0 ? val : 0;
}

inline double Fpn_minus(double p1, double n, double M, double b, double L, double phi, double mu) {
    double E1 = std::sqrt(M*M + p1*p1);
    double val = (mu - std::sqrt((E1 - b)*(E1 - b) + std::pow(2 * M_PI / L * (n + phi), 2))) / (2 * M_PI);
    return val > 0 ? val : 0;
}

// Интегрирование по p1 для одного n
// Простая формула прямоугольников
inline double integrate_p1_plus(double n, double M, double b, double L, double phi, double mu, int N_h, double p_left, double p_right) {
    double sum = 0.0;
    double step = (p_right - p_left) / N_h;
    for (int i = 0; i < N_h; ++i) {
        double p1 = p_left + (i + 0.5) * step;
        sum += Fpn_plus(p1, n, M, b, L, phi, mu) * step;
    }
    return sum;
}

inline double integrate_p1_minus(double n, double M, double b, double L, double phi, double mu, int N_h, double p_left, double p_right) {
    double sum = 0.0;
    double step = (p_right - p_left) / N_h;
    for (int i = 0; i < N_h; ++i) {
        double p1 = p_left + (i + 0.5) * step;
        sum += Fpn_minus(p1, n, M, b, L, phi, mu) * step;
    }
    return sum;
}




// --- Перенос логики Omega_mu_L.py ---
int n_max_plus(double M, double b, double L, double mu, double phi) {
    if (std::abs(b) > M && b < 0) {
        double test = (mu - std::sqrt((std::sqrt(b * b - M * M) + b) * (std::sqrt(b * b - M * M) + b) + std::pow(2 * M_PI / L * (0 + phi), 2))) / (2 * M_PI);
        if (test <= 0) return -1;
        else return int(L * mu / (2 * M_PI) - phi);
    } else if (mu * mu <= (M + b) * (M + b)) {
        return -1;
    } else {
        return int(L / (2 * M_PI) * std::sqrt(mu * mu - std::pow(M + b, 2)) - phi);
    }
}

int n_max_minus(double M, double b, double L, double mu, double phi) {
    if (std::abs(b) > M && b > 0) {
        double test = (mu - std::sqrt((std::sqrt(b * b - M * M) - b) * (std::sqrt(b * b - M * M) - b) + std::pow(2 * M_PI / L * (0 + phi), 2))) / (2 * M_PI);
        if (test <= 0) return -1;
        else return int(L * mu / (2 * M_PI) - phi);
    } else if (mu * mu <= (M - b) * (M - b)) {
        return -1;
    } else {
        return int(L / (2 * M_PI) * std::sqrt(mu * mu - std::pow(M - b, 2)) - phi);
    }
}

void integration_limits_plus(double M, double b, double mu, double L, int Nmax, double phi, std::vector<double>& p_left, std::vector<double>& p_right) {
    p_left.resize(Nmax + 1);
    p_right.resize(Nmax + 1);
    for (int i = 0; i <= Nmax; ++i) {
        double n = i;
        double SQ_arg = mu * mu - 4 * std::pow(M_PI / L, 2) * std::pow(n + phi, 2);
        double SQ = SQ_arg > 0 ? std::sqrt(SQ_arg) : 0.0;
        p_left[i] = 0.0;
        double right_arg = 0.0;
        if (b > 0) {
            right_arg = std::pow(-b - SQ, 2) - M * M;
        } else if (b < 0) {
            right_arg = std::pow(-b + SQ, 2) - M * M;
        } else {
            right_arg = SQ * SQ - M * M;
        }
        p_right[i] = right_arg > 0 ? std::sqrt(right_arg) : 0.0;
    }
}

void integration_limits_minus(double M, double b, double mu, double L, int Nmax, double phi, std::vector<double>& p_left, std::vector<double>& p_right) {
    p_left.resize(Nmax + 1);
    p_right.resize(Nmax + 1);
    for (int i = 0; i <= Nmax; ++i) {
        double n = i;
        double SQ_arg = mu * mu - 4 * std::pow(M_PI / L, 2) * std::pow(n + phi, 2);
        double SQ = SQ_arg > 0 ? std::sqrt(SQ_arg) : 0.0;
        p_left[i] = 0.0;
        double right_arg = 0.0;
        if (b > 0) {
            right_arg = std::pow(b + SQ, 2) - M * M;
        } else if (b < 0) {
            right_arg = std::pow(b - SQ, 2) - M * M;
        } else {
            right_arg = SQ * SQ - M * M;
        }
        p_right[i] = right_arg > 0 ? std::sqrt(right_arg) : 0.0;
    }
}

double fun_Omega_L_mu_int(double mu, double L, double b, double M, double phi = 0, int N_h = 100) {
    int Nmax_plus = n_max_plus(M, b, L, mu, phi);
    double Unp = 0.0;
    if (Nmax_plus >= 0) {
        std::vector<double> p_left, p_right;
        integration_limits_plus(M, b, mu, L, Nmax_plus, phi, p_left, p_right);
        for (int i = 0; i <= Nmax_plus; ++i) {
            double n = i;
            double step = (p_right[i] - p_left[i]) / N_h;
            for (int j = 0; j < N_h; ++j) {
                double p1 = p_left[i] + (j + 0.5) * step;
                Unp += Fpn_plus(p1, n, M, b, L, phi, mu) * step;
            }
        }
    }
    int Nmax_minus = n_max_minus(M, b, L, mu, phi);
    double Unm = 0.0;
    if (Nmax_minus >= 0) {
        std::vector<double> p_left, p_right;
        integration_limits_minus(M, b, mu, L, Nmax_minus, phi, p_left, p_right);
        for (int i = 0; i <= Nmax_minus; ++i) {
            double n = i;
            double step = (p_right[i] - p_left[i]) / N_h;
            for (int j = 0; j < N_h; ++j) {
                double p1 = p_left[i] + (j + 0.5) * step;
                Unm += Fpn_minus(p1, n, M, b, L, phi, mu) * step;
            }
        }
    }
    return -(2.0 / L) * (Unp + Unm);
}


// --- Аналог функции midpoints из utils.py ---
std::vector<double> midpoints(double a, double b, int N) {
    std::vector<double> result;
    double step = (b - a) / N;
    for (int i = 0; i < N; ++i)
        result.push_back(a + (i + 0.5) * step);
    return result;
}

// --- Аналог dU_int из dU.py ---
double dU_int(double u, double phi, double b, double M) {
    double p = u / (1.0 - u);
    double term1 = M * M / p;
    double term2 = std::sqrt(b * b + p * p + 2 * b * p * std::cos(phi));
    double term3 = std::sqrt(b * b + p * p - 2 * b * p * std::cos(phi));
    double term4 = std::sqrt(M * M + b * b + p * p + 2 * b * std::sqrt(M * M + p * p * std::pow(std::cos(phi), 2)));
    double term5 = std::sqrt(M * M + b * b + p * p - 2 * b * std::sqrt(M * M + p * p * std::pow(std::cos(phi), 2)));
    return -(-term1 - term2 - term3 + term4 + term5) * p / (M_PI * M_PI) / std::pow(1.0 - u, 2);
}

// --- Аналог Omega_L_int из Omega_L.py ---
double Omega_L_int(double u1, double u3, double L, double b, double M, double phi = 0) {
    double p1 = u1 / (1.0 - u1);
    double p3 = u3 / (1.0 - u3);
    double E1 = std::sqrt(M * M + p1 * p1);
    double B_plus = std::sqrt(p3 * p3 + std::pow(E1 + b, 2));
    double B_minus = std::sqrt(p3 * p3 + std::pow(E1 - b, 2));
    double term1 = 1 - 2 * std::cos(2 * M_PI * phi) * std::exp(-L * B_plus) + std::exp(-2 * L * B_plus);
    double term2 = 1 - 2 * std::cos(2 * M_PI * phi) * std::exp(-L * B_minus) + std::exp(-2 * L * B_minus);
    return -4 * std::log(term1 * term2) / std::pow(2 * M_PI, 2) / std::pow(1.0 - u1, 2) / std::pow(1.0 - u3, 2) / L;
}


// --- Основной расчёт ---
int main() {
    // Параметры сетки
    int MU_SPLIT = 20, L_SPLIT = 20, B_SPLIT = 21, M_SPLIT = 20;
    double mu_min = 2.5, mu_max = 4.0;
    double L_min = 0.5, L_max = 3.0;
    double b_min = -1.0, b_max = 1.0;
    double M_min = 0.0, M_max = 5.0;
    double phi = 0.0;
    int N_h_p1 = 100, N_h_p2 = 100, N_h_p = 100, N_h_phi = 100;

    // Генерация сеток
    std::vector<double> mu_vals, L_vals, b_vals, M_vals;
    for (int i = 0; i < MU_SPLIT; ++i) mu_vals.push_back(mu_min + i * (mu_max - mu_min) / (MU_SPLIT - 1));
    for (int i = 0; i < L_SPLIT; ++i) L_vals.push_back(L_min + i * (L_max - L_min) / (L_SPLIT - 1));
    for (int i = 0; i < B_SPLIT; ++i) b_vals.push_back(b_min + i * (b_max - b_min) / (B_SPLIT - 1));
    for (int i = 0; i < M_SPLIT; ++i) M_vals.push_back(M_min + i * (M_max - M_min) / (M_SPLIT - 1));

    // --- Предварительные расчёты Omega_L_phys ---
    std::vector< std::vector< std::vector< std::vector<double> > > > Omega_L_phys(
        L_SPLIT,
        std::vector< std::vector< std::vector<double> > >(B_SPLIT,
            std::vector< std::vector<double> >(M_SPLIT,
                std::vector<double>(1, 0.0))));
    for (int L_ind = 0; L_ind < L_SPLIT; ++L_ind) {
        for (int b_ind = 0; b_ind < B_SPLIT; ++b_ind) {
            for (int M_ind = 0; M_ind < M_SPLIT; ++M_ind) {
                double sum = 0.0;
                for (int i = 0; i < N_h_p1; ++i) {
                    double u1 = double(i + 0.5) / N_h_p1;
                    for (int j = 0; j < N_h_p2; ++j) {
                        double u3 = double(j + 0.5) / N_h_p2;
                        sum += Omega_L_int(u1, u3, L_vals[L_ind], b_vals[b_ind], M_vals[M_ind], phi);
                    }
                }
                double dp1 = 1.0 / N_h_p1;
                double dp2 = 1.0 / N_h_p2;
                Omega_L_phys[L_ind][b_ind][M_ind][0] = sum * dp1 * dp2;
            }
        }
    }

    // --- Предварительные расчёты dU_phys ---
    std::vector< std::vector<double> > dU_phys(B_SPLIT, std::vector<double>(M_SPLIT, 0.0));
    for (int b_ind = 0; b_ind < B_SPLIT; ++b_ind) {
        for (int M_ind = 0; M_ind < M_SPLIT; ++M_ind) {
            double sum = 0.0;
            for (int i = 0; i < N_h_p; ++i) {
                double u = double(i + 0.5) / N_h_p;
                for (int j = 0; j < N_h_phi; ++j) {
                    double phi_loc = (M_PI / 2) * double(j + 0.5) / N_h_phi;
                    sum += dU_int(u, phi_loc, b_vals[b_ind], M_vals[M_ind]);
                }
            }
            double dp = 1.0 / N_h_p;
            double dphi = (M_PI / 2) / N_h_phi;
            dU_phys[b_ind][M_ind] = sum * dp * dphi;
        }
    }

    // --- Основной расчёт Omega_ren_phys ---
    std::vector< std::vector< std::vector< std::vector<double> > > > Omega_ren_phys(
        MU_SPLIT,
        std::vector< std::vector< std::vector<double> > >(L_SPLIT,
            std::vector< std::vector<double> >(B_SPLIT,
                std::vector<double>(M_SPLIT, 0.0))));

    clock_t start_time = clock();
    for (int mu_ind = 0; mu_ind < MU_SPLIT; ++mu_ind) {
        for (int L_ind = 0; L_ind < L_SPLIT; ++L_ind) {
            for (int b_ind = 0; b_ind < B_SPLIT; ++b_ind) {
                for (int M_ind = 0; M_ind < M_SPLIT; ++M_ind) {
                    double mu = mu_vals[mu_ind];
                    double L = L_vals[L_ind];
                    double b = b_vals[b_ind];
                    double M = M_vals[M_ind];
                    double Omega_mu_L = fun_Omega_L_mu_int(mu, L, b, M, phi, 100)
                        - fun_Omega_L_mu_int(mu, L, b, 0, phi, 100)
                        + fun_Omega_L_mu_int(mu, L, 0, 0, phi, 100);
                    double Omega_ren = (M * M / (2.0 * 1.0) +
                        dU_phys[b_ind][M_ind] +
                        Omega_L_phys[L_ind][b_ind][M_ind][0] +
                        Omega_mu_L);
                    Omega_ren_phys[mu_ind][L_ind][b_ind][M_ind] = Omega_ren;
                }
            }
        }
    }
    clock_t end_time = clock();
    double elapsed_sec = double(end_time - start_time) / CLOCKS_PER_SEC;

    // Сохранение результата
    std::ofstream fout("Omega_ren_phys_cpp.txt");
    for (int mu_ind = 0; mu_ind < MU_SPLIT; ++mu_ind) {
        for (int L_ind = 0; L_ind < L_SPLIT; ++L_ind) {
            for (int b_ind = 0; b_ind < B_SPLIT; ++b_ind) {
                for (int M_ind = 0; M_ind < M_SPLIT; ++M_ind) {
                    fout << Omega_ren_phys[mu_ind][L_ind][b_ind][M_ind] << " ";
                }
                fout << std::endl;
            }
            fout << std::endl;
        }
        fout << std::endl;
    }
    fout.close();
    std::cout << "Расчёт завершён. Результаты сохранены в Omega_ren_phys_cpp.txt" << std::endl;
    std::cout << "Время выполнения: " << elapsed_sec << " секунд" << std::endl;
    return 0;
}
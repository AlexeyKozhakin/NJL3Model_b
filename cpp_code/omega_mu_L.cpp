#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

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

// Пример функции аналогичной fun_Omega_L_mu_int
// Здесь упрощённая версия без сложных границ интегрирования
// Для реального переноса нужно реализовать n_max_plus, n_max_minus и integration_limits

double fun_Omega_L_mu_int(double mu, double L, double b, double M, double phi = 0, int N_h = 100) {
    int Nmax = 10; // пример, в реальном коде вычисляется
    double result = 0.0;
    for (int n = 0; n <= Nmax; ++n) {
        double p_left = 0.0; // пример
        double p_right = 1.0; // пример
        result += integrate_p1_plus(n, M, b, L, phi, mu, N_h, p_left, p_right);
        result += integrate_p1_minus(n, M, b, L, phi, mu, N_h, p_left, p_right);
    }
    return -(2.0 / L) * result;
}

int main() {
    // Замер времени
    clock_t start_time = clock();

    // Количество точек для каждого параметра
    int MU_SPLIT = 10;
    int L_SPLIT = 10;
    int B_SPLIT = 11;
    int M_SPLIT = 10;

    // Диапазоны
    double mu_min = 2.5, mu_max = 4.0;
    double L_min = 0.1, L_max = 3.0;
    double b_min = -1.0, b_max = 1.0;
    double M_min = 0.0, M_max = 5.0;

    // Генерация равномерных сеток
    std::vector<double> mu_vals;
    std::vector<double> L_vals;
    std::vector<double> b_vals;
    std::vector<double> M_vals;
    for (int i = 0; i < MU_SPLIT; ++i)
        mu_vals.push_back(mu_min + i * (mu_max - mu_min) / (MU_SPLIT - 1));
    for (int i = 0; i < L_SPLIT; ++i)
        L_vals.push_back(L_min + i * (L_max - L_min) / (L_SPLIT - 1));
    for (int i = 0; i < B_SPLIT; ++i)
        b_vals.push_back(b_min + i * (b_max - b_min) / (B_SPLIT - 1));
    for (int i = 0; i < M_SPLIT; ++i)
        M_vals.push_back(M_min + i * (M_max - M_min) / (M_SPLIT - 1));

    double phi = 0.0;
    int N_h = 100;

    // 4D массив результатов
    std::vector< std::vector< std::vector< std::vector<double> > > > Omega_ren_phys(
        mu_vals.size(),
        std::vector< std::vector< std::vector<double> > >(L_vals.size(),
            std::vector< std::vector<double> >(b_vals.size(),
                std::vector<double>(M_vals.size(), 0.0))));

    // Последовательный расчёт
    for (size_t mu_ind = 0; mu_ind < mu_vals.size(); ++mu_ind) {
        for (size_t L_ind = 0; L_ind < L_vals.size(); ++L_ind) {
            for (size_t b_ind = 0; b_ind < b_vals.size(); ++b_ind) {
                for (size_t M_ind = 0; M_ind < M_vals.size(); ++M_ind) {
                    double mu = mu_vals[mu_ind];
                    double L = L_vals[L_ind];
                    double b = b_vals[b_ind];
                    double M = M_vals[M_ind];
                    Omega_ren_phys[mu_ind][L_ind][b_ind][M_ind] = fun_Omega_L_mu_int(mu, L, b, M, phi, N_h);
                }
            }
        }
    }

    // Сохранение результата в файл
    std::ofstream fout("Omega_ren_phys.txt");
    for (size_t mu_ind = 0; mu_ind < mu_vals.size(); ++mu_ind) {
        for (size_t L_ind = 0; L_ind < L_vals.size(); ++L_ind) {
            for (size_t b_ind = 0; b_ind < b_vals.size(); ++b_ind) {
                for (size_t M_ind = 0; M_ind < M_vals.size(); ++M_ind) {
                    fout << Omega_ren_phys[mu_ind][L_ind][b_ind][M_ind] << " ";
                }
                fout << std::endl;
            }
            fout << std::endl;
        }
        fout << std::endl;
    }
    fout.close();
    clock_t end_time = clock();
    double elapsed_sec = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Расчёт завершён. Результаты сохранены в Omega_ren_phys.txt" << std::endl;
    std::cout << "Время выполнения: " << elapsed_sec << " секунд" << std::endl;
    return 0;
}

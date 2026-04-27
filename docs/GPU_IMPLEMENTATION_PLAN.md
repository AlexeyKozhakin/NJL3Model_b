# GPU Implementation Plan

## Цель

Переписать вычисление `Omega_ren(M, b, mu, L)` на **PyTorch + GPU** с полной векторизацией по всем осям (`mu`, `L`, `M`, `b`).

**Результат:** фазовая диаграмма 50×50 (mu, L) за **~1 минуту** вместо часов.

---

## Архитектура (высокий уровень)

### Структура файлов

```
functions/
├── dU_torch.py          # dU(b, M) на PyTorch
├── Omega_L_torch.py     # Omega_L(L, b, M) на PyTorch
├── Omega_muL_torch.py   # Omega_muL(mu, L, b, M) — двухуровневый батчинг
└── Omega_ren_torch.py   # Сборка + поиск минимума

notebooks/
└── 40_gpu_phase_diagram.py   # Главный скрипт

cache/                   # Промежуточные результаты (сохраняются на диск)
├── dU.pt
├── Omega_L.pt
└── Omega_muL.pt
```

### Поток вычислений

```
КОНФИГ: mu[N_MU], L[N_L], b[N_B], M[N_M], g, device, dtype, chunk_size C

         │
         ▼
┌─────────────────┐   не зависит от mu       сохранить
│  dU(b, M)       │   1 проход, пик ~100 MB → cache/dU.pt
│  shape(N_B,N_M) │
└─────────────────┘
         │
         ▼
┌─────────────────────┐   цикл по M (N_M итераций)    сохранить
│  Omega_L(L, b, M)   │   пик ~200 MB за шаг        → cache/Omega_L.pt
│  shape(N_L,N_B,N_M) │
└─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Omega_muL(mu, L, b, M)                         │
│  shape (N_MU, N_L, N_B, N_M)                   │
│                                                 │
│  for mu_i in range(N_MU):                      │  пик ~750 MB
│    for (b_chunk, M_chunk) размером C×C:        │  константа
│      F = clamp(Fpn, min=0)   ← ReLU            │  при любом
│      маска по n > n_max                        │  N_B × N_M
│      сумма по (n, p)                           │
└─────────────────────────────────────────────────┘
         │                  сохранить → cache/Omega_muL.pt
         ▼
┌──────────────────────────────────────────────────────┐
│  Omega_ren = tree(M) + dU + Omega_L + Omega_muL      │
│                                                      │
│  broadcast:                                          │
│    tree     : (      N_M) → (1,   1,   1, N_M)      │
│    dU       : (  N_B,N_M) → (1,   1, N_B, N_M)      │
│    Omega_L  : (N_L,N_B,N_M) → (1, N_L, N_B, N_M)   │
│    Omega_muL: (N_MU, N_L, N_B, N_M)                 │
│                                                      │
│  shape результата: (N_MU, N_L, N_B, N_M) ≈ 25 MB    │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  argmin по (b, M)                │
│  для каждой точки (mu_i, L_j)   │
│  → M_min[N_MU,N_L], b_min[...] │
└──────────────────────────────────┘
         │
         ▼
    Фазовая диаграмма

```

### Ключевые принципы дизайна

**1. Разделение по зависимостям.**
`dU` и `Omega_L` не зависят от `mu` — вычисляются один раз и кэшируются на диск.
При смене `mu`-сетки пересчитывается только `Omega_muL`.

**2. Двухуровневый батчинг в `Omega_muL`.**
Внешний цикл по `mu`, внутренний по чанкам `(b, M)` размера `C×C`.
Пиковая память ~750 MB — константа при любом размере сетки:

| Сетка (b×M) | Пик GPU | Omega_muL.pt |
|---|---|---|
| 50×50   | ~750 MB | 25 MB  |
| 100×100 | ~750 MB | 100 MB |
| 200×200 | ~750 MB | 400 MB |
| 500×500 | ~750 MB | 2.5 GB |

**3. ReLU вместо явных границ интегрирования.**
`F = clamp(Fpn, min=0)` — никакого бранчинга на GPU.
Двухкорневой случай обрабатывается автоматически.

**4. Единая формула ренормализации для всех функций.**
```
f_phys(b, M) = f(b, M) - f(b, 0) + f(0, 0)
```
Применяется одинаково к `dU`, `Omega_L`, `Omega_muL`.

**5. `dtype` и `device` как параметры конфига.**
`float32` по умолчанию (скорость), `float64` как fallback (точность).
Переключается одной строкой.

---

## Параметры и конфигурация

```python
import torch

# Размеры сеток
N_MU = 50
N_L = 50
N_M = 50
N_B = 50

# Точность: float32 (быстро) или float64 (точно)
# Переключается параметром dtype
dtype = torch.float32  # или torch.float64

# Устройство: cuda (GPU) или cpu (fallback)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Сетки (равномерные)
mu = torch.linspace(0.1, 7.0, N_MU, dtype=dtype, device=device)
L  = torch.linspace(0.1, 4.0, N_L,  dtype=dtype, device=device)
M  = torch.linspace(0.0, 10.0, N_M, dtype=dtype, device=device)
b  = torch.linspace(0.0, 10.0, N_B, dtype=dtype, device=device)

# Параметр модели
g = -1.0

# Размер интегральной сетки
N_P = 100   # для p, p1, p2
N_PHI = 100 # для phi
```

---

## Память GPU (лимит 15 GB)

| Тензор | Размер (элементов) | float32 | float64 |
|--------|-------------------|---------|---------|
| `dU` (cache) | 50×50 = 2.5K | 10 KB | 20 KB |
| `Omega_L` (cache) | 50×50×50 = 125K | 500 KB | 1 MB |
| `Omega_muL` (cache) | 50×50×50×50 = 6.25M | 25 MB | 50 MB |
| `Omega_ren` (временный) | 50×50×50×50 = 6.25M | 25 MB | 50 MB |
| **Промежуточные (батч)** | ~1M × 20 × 100 | ~800 MB | ~1.6 GB |
| **Итого + запас** | | **~2–3 GB** | **~4–6 GB** |

**Вывод:** Лимит 15 GB позволяет использовать **батчи до 1–1.5M точек** без проблем.

---

## Функция 1: tree(M)

### Формула
$$\text{tree}(M) = \frac{M^2}{2g}$$

### Вход
- `M`: тензор размера `(N_M,)` или любой broadcast-формы

### Выход
- `tree`: тензор той же формы

### Реализация (PyTorch)
```python
def compute_tree(M, g):
    return M**2 / (2 * g)
```

### Сложность
- O(N) элементарных операций
- Время: мгновенно

---

## Функция 2: dU_phys(b, M)

### Формула

Замена переменной: $p = \frac{u}{1-u}$, где $u \in [0, 1)$, $p \in [0, \infty)$.

$$\Delta U_{\text{phys}}(b, M) = \int_0^1 du \int_0^{\pi/2} d\phi \; F_{dU}(u, \phi, b, M)$$

Интегранд:

$$F_{dU}(u, \phi, b, M) = -\frac{p}{(1-u)^2 \pi^2} \left( -\frac{M^2}{p} - R_2 - R_3 + R_4 + R_5 \right)$$

gде $p = \frac{u}{1-u}$ и:

$$
\begin{aligned}
R_2 &= \sqrt{b^2 + p^2 + 2bp\cos\phi} \\
R_3 &= \sqrt{b^2 + p^2 - 2bp\cos\phi} \\
R_4 &= \sqrt{M^2 + b^2 + p^2 + 2b\sqrt{M^2 + p^2\cos^2\phi}} \\
R_5 &= \sqrt{M^2 + b^2 + p^2 - 2b\sqrt{M^2 + p^2\cos^2\phi}}
\end{aligned}
$$

### Вход
- `b`: тензор `(N_B,)`
- `M`: тензор `(N_M,)`
- Сетки: `u[N_P]`, `phi[N_PHI]`

### Выход
- `dU`: тензор `(N_B, N_M)`

### Алгоритм
```
1. Создать meshgrid: (b, M, u, phi) → 4D тензор (N_B, N_M, N_P, N_PHI)
2. p = u / (1 - u)
3. Вычислить интегранд F(b, M, p, phi) — векторизованно
4. Суммировать по осям u и phi: dU = sum(F) * du * dphi
```

### Псевдокод (PyTorch)
```python
def compute_dU(b, M, N_p=100, N_phi=100):
    # Сетки интегрирования
    u = torch.linspace(0, 1, N_p+1, dtype=b.dtype, device=b.device)
    phi = torch.linspace(0, torch.pi/2, N_phi+1, dtype=b.dtype, device=b.device)
    
    # Середины отрезков (метод средних)
    u_mid = (u[:-1] + u[1:]) / 2
    phi_mid = (phi[:-1] + phi[1:]) / 2
    du = u[1] - u[0]
    dphi = phi[1] - phi[0]
    
    # Meshgrid: (N_B, N_M, N_p, N_phi)
    b_grid, M_grid, u_grid, phi_grid = torch.meshgrid(b, M, u_mid, phi_mid, indexing='ij')
    
    p = u_grid / (1 - u_grid)
    
    # Вычисление интегранда F_dU(p, phi, b, M) — см. формулу выше
    # Векторизованно по всем осям (N_B, N_M, N_p, N_phi)
    F = dU_integrand(p, phi_grid, b_grid, M_grid)
    
    # Интегрирование
    dU = torch.sum(F, dim=(2, 3)) * du * dphi
    return dU
```

### Сложность
- O(N_B × N_M × N_p × N_phi)
- Время GPU: < 1 сек

---

## Функция 3: Omega_L(L, b, M)

### Формула

Ренормализация:

$$\Omega_L^{\text{phys}}(L, b, M) = \Omega_L(L, b, M) - \Omega_L(L, b, 0) + \Omega_L(L, 0, 0)$$

Базовый интеграл (двумерный, $\phi = 0$):

$$\Omega_L(L, b, M) = \int_0^1 du_1 \int_0^1 du_3 \; F_L(u_1, u_3, L, b, M)$$

Замены: $p_1 = \frac{u_1}{1-u_1}$, $p_3 = \frac{u_3}{1-u_3}$.

$$F_L(u_1, u_3, L, b, M) = -\frac{4}{(2\pi)^2 (1-u_1)^2 (1-u_3)^2 L} \cdot \ln\left[ \mathcal{T}_+ \cdot \mathcal{T}_- \right]$$

gде:

$$
\begin{aligned}
E_1 &= \sqrt{M^2 + p_1^2} \\
B_{\pm} &= \sqrt{p_3^2 + (E_1 \pm b)^2} \\
\mathcal{T}_{\pm} &= 1 - 2\cos(2\pi\phi) \cdot e^{-LB_{\pm}} + e^{-2LB_{\pm}}
\end{aligned}
$$

(в физике модели $\phi = 0$, поэтому $\cos(2\pi\phi) = 1$).

### Вход
- `L`: тензор `(N_L,)`
- `b`: тензор `(N_B,)`
- `M`: тензор `(N_M,)`

### Выход
- `Omega_L_phys`: тензор `(N_L, N_B, N_M)`

### Алгоритм
```
1. Для каждой комбинации (L, b, M) вычислить Omega_L(L, b, M)
   - meshgrid (L, b, M, u1, u3) → 5D тензор
   - p1 = u1/(1-u1), p3 = u3/(1-u3)
   - Вычислить интегранд (векторизованно)
   - Суммировать по u1, u3
2. Вычислить Omega_L(L, b, 0) — то же с M=0
3. Вычислить Omega_L(L, 0, 0) — то же с b=0, M=0
4. Omega_L_phys = #1 - #2 + #3
```

### Псевдокод
```python
def compute_Omega_L(L, b, M, N_p=100):
    # 3 вызова: (L,b,M), (L,b,0), (L,0,0)
    # Каждый — 2D интеграл по (u1, u3)
    
    def Omega_L_single(L_val, b_val, M_val):
        # meshgrid (N_L, N_B, N_M, N_p, N_p)
        # Интегранд F_L(u1, u3, L, b, M) — см. формулу выше
        return integral
    
    term1 = Omega_L_single(L, b, M)      # (N_L, N_B, N_M)
    term2 = Omega_L_single(L, b, torch.zeros_like(M))  # (N_L, N_B, 1) → broadcast
    term3 = Omega_L_single(L, torch.zeros_like(b), torch.zeros_like(M))  # (N_L, 1, 1)
    
    return term1 - term2 + term3
```

### Сложность
- O(N_L × N_B × N_M × N_p²)
- Время GPU: ~1–2 сек

---

## Функция 4: Omega_mu_L(mu, L, b, M) — СЛОЖНАЯ

### Формула

Ренормализация:

$$\Omega_{\mu L}^{\text{phys}} = \Omega_{\mu L}(\mu, L, b, M) - \Omega_{\mu L}(\mu, L, b, 0) + \Omega_{\mu L}(\mu, L, 0, 0)$$

Базовый вклад (сумма по модам $n$ и интеграл по $p$):

$$\Omega_{\mu L}(\mu, L, b, M) = -\frac{2}{L} \sum_{n=0}^{n_{\max}^{+}} \left[ C_n \cdot I_n^{+} \right] - \frac{2}{L} \sum_{n=0}^{n_{\max}^{-}} \left[ C_n \cdot I_n^{-} \right]$$

gде $C_n = 1$ при $n=0$ и $C_n = 2$ при $n > 0$ (учёт $n \leftrightarrow -n$ симметрии).

#### Функции $F_{\pm}$ (подынтегральные)

$$E_1 = \sqrt{M^2 + p^2}$$

$$F_{+}(p, n, M, b, L, \mu) = \max\left(0, \; \frac{\mu - \sqrt{(E_1 + b)^2 + \left(\frac{2\pi}{L}(n+\phi)\right)^2}}{2\pi} \right)$$

$$F_{-}(p, n, M, b, L, \mu) = \max\left(0, \; \frac{\mu - \sqrt{(E_1 - b)^2 + \left(\frac{2\pi}{L}(n+\phi)\right)^2}}{2\pi} \right)$$

(в физике модели $\phi = 0$).

#### Пределы интегрирования $I_n^{\pm}$

**Обычный случай (1 корень):**

$$S_n = \sqrt{\mu^2 - \left(\frac{2\pi n}{L}\right)^2}$$

Для $F_{+}$:
- Если $b \geq 0$: $p_{\min} = 0$, $p_{\max} = \sqrt{(S_n - b)^2 - M^2}$ (при $S_n > b + M$, иначе вклад 0)
- Если $b < 0$: $p_{\min} = 0$, $p_{\max} = \sqrt{(S_n + |b|)^2 - M^2}$ (при $S_n > |b| + M$, иначе вклад 0)

Для $F_{-}$:
- Если $b \leq 0$: $p_{\min} = 0$, $p_{\max} = \sqrt{(S_n - |b|)^2 - M^2}$ (при $S_n > |b| + M$, иначе вклад 0)
- Если $b > 0$: $p_{\min} = 0$, $p_{\max} = \sqrt{(S_n - b)^2 - M^2}$ (при $S_n > b + M$, иначе вклад 0)

**Двухкорневой случай (два корня):**

Возникает когда $|b| > M$ и функция в $p=0$ отрицательна:
- Для $F_{+}$: при $b < 0$ и $F_{+}(0) < 0$
- Для $F_{-}$: при $b > 0$ и $F_{-}(0) < 0$

В этом случае:

$$p_{\min} = \sqrt{(|b| - S_n)^2 - M^2}, \quad p_{\max} = \sqrt{(|b| + S_n)^2 - M^2}$$

(при условии, что $p_{\min} < p_{\max}$ и $S_n > 0$).

#### $n_{\max}$ — определение

$$n_{\max}^{+}(M, b, L, \mu) = \left\lfloor \frac{L}{2\pi} \sqrt{\max(0, \mu^2 - (M+b)^2)} \right\rfloor$$

$$n_{\max}^{-}(M, b, L, \mu) = \left\lfloor \frac{L}{2\pi} \sqrt{\max(0, \mu^2 - (M-b)^2)} \right\rfloor$$

**Особые случаи:**
- Если $\mu^2 \leq (M+b)^2$ для $F_{+}$: $n_{\max}^{+} = -1$ (нет вклада)
- Если $\mu^2 \leq (M-b)^2$ для $F_{-}$: $n_{\max}^{-} = -1$ (нет вклада)
- Если $|b| > M$ и $F_{\pm}(0) \leq 0$: $n_{\max} = -1$ (нет вклада)
- Иначе: $n_{\max} = \left\lfloor \frac{L\mu}{2\pi} \right\rfloor$

### Проблема: n_max разный для каждой точки

Для разных `(mu, L, b, M)` значение `n_max` отличается:
- При `mu=7, L=0.1`: `n_max` ~ 10–15
- При `mu=1, L=4.0`: `n_max` = -1 (нет вклада)

### Решение: Padding + Masking + ReLU (без явных границ)

> ✅ **Решено:** вместо вычисления `p_left/p_right` и обработки двухкорневого случая
> использовать `F.clamp(min=0)` (= ReLU). Это полностью устраняет бранчинг на GPU.
> Двухкорневой случай обрабатывается автоматически: ReLU зануляет области где `F < 0`,
> оставляя только реальную Ферми-область (кольцо или диск).
> Прецедент: `Fpn_plus_numpy` в `Omega_mu_L.py` уже делает это через `np.heaviside`.

> ✅ **Двухуровневый батчинг** — делает архитектуру масштабируемой до любого N_B × N_M:
> - Внешний цикл по `mu` (N_MU итераций)
> - Внутренний цикл по чанкам `(b, M)` фиксированного размера C×C (например 50×50 = 2500 точек)
> - Пиковая память на шаге: `3 × N_L × C² × (n_max+1) × N_p × 4 bytes ≈ 750 MB` — константа,
>   не зависящая от размера сетки. Сетку 500×500 можно считать теми же 750 MB, просто дольше.

```
Цикл по mu (N_MU итераций):
  Цикл по (b, M) чанкам размером C×C:
    тензор (N_L, C_b, C_M, n_max+1, N_p)  ← фиксированная память
    F_plus  = clamp((mu - sqrt((E1+b)² + (2πn/L)²)) / 2π, min=0)
    F_minus = clamp((mu - sqrt((E1-b)² + (2πn/L)²)) / 2π, min=0)
    маска по n: зануляем строки n > n_max[i]
    суммировать по p (× dp) и по n (с коэффициентом C_n)
    записать результат в Omega_muL[mu_idx, :, b_chunk, M_chunk]
```

### Вход
- `mu, L, b, M`: тензоры произвольной формы (flatten → батч)

### Выход
- `Omega_muL`: тензор той же формы

### Псевдокод
```python
def compute_Omega_muL_batch(mu_batch, L_batch, b_batch, M_batch, N_p=100):
    batch_size = mu_batch.numel()
    
    # Шаг 1: вычислить n_max для каждой точки
    n_max = compute_n_max_vectorized(mu_batch, L_batch, b_batch, M_batch)
    n_max_global = int(n_max.max().item())
    
    if n_max_global < 0:
        return torch.zeros_like(mu_batch)
    
    # Шаг 2: сетка p
    p = torch.linspace(0, p_max, N_p, dtype=mu_batch.dtype, device=mu_batch.device)
    dp = p[1] - p[0]
    
    # Шаг 3: тензор (batch, n_max_global, N_p)
    # Meshgrid: (batch, n_max_global, N_p)
    n_range = torch.arange(n_max_global + 1, device=mu_batch.device)
    
    mu_grid = mu_batch.view(-1, 1, 1)
    L_grid = L_batch.view(-1, 1, 1)
    b_grid = b_batch.view(-1, 1, 1)
    M_grid = M_batch.view(-1, 1, 1)
    n_grid = n_range.view(1, -1, 1)
    p_grid = p.view(1, 1, -1)
    
    # Шаг 4: вычислить F_plus и F_minus (векторизованно)
    # F_plus(p, n, M, b, L, mu) и F_minus(...)
    F = compute_F_vectorized(p_grid, n_grid, M_grid, b_grid, L_grid, mu_grid)
    
    # Шаг 5: маска для n > n_max
    mask = (n_grid <= n_max.view(-1, 1, 1))  # (batch, n_max_global, 1)
    F = F * mask  # зануляем лишние n
    
    # Шаг 6: интегрирование и суммирование
    integral = torch.sum(F, dim=2) * dp  # (batch, n_max_global)
    result = torch.sum(integral, dim=1)  # (batch,)
    
    return -2.0 / L_batch * result

# Полная функция с ренормализацией
def compute_Omega_muL_full(mu, L, b, M):
    # Flatten → батчи
    shape = mu.shape
    mu_f = mu.flatten()
    L_f = L.flatten()
    b_f = b.flatten()
    M_f = M.flatten()
    
    batch_size = mu_f.numel()
    results = []
    
    for i in range(0, batch_size, BATCH_SIZE):
        end = min(i + BATCH_SIZE, batch_size)
        term1 = compute_Omega_muL_batch(mu_f[i:end], L_f[i:end], b_f[i:end], M_f[i:end])
        term2 = compute_Omega_muL_batch(mu_f[i:end], L_f[i:end], b_f[i:end], torch.zeros_like(M_f[i:end]))
        term3 = compute_Omega_muL_batch(mu_f[i:end], L_f[i:end], torch.zeros_like(b_f[i:end]), torch.zeros_like(M_f[i:end]))
        results.append(term1 - term2 + term3)
    
    return torch.cat(results).reshape(shape)
```

### Сложность
- O(batch × n_max × N_p) на батч
- Время GPU: ~10–30 сек на всю сетку

---

## Функция 5: Сборка Omega_ren

### Формула
$$\Omega_{\text{ren}} = \text{tree}(M) + dU(b, M) + \Omega_L^{\text{phys}}(L, b, M) + \Omega_{\mu L}^{\text{phys}}(\mu, L, b, M)$$

### Вход
- Кэшированные тензоры: `tree`, `dU`, `Omega_L`, `Omega_muL`

### Выход
- `Omega_ren[mu, L, b, M]`: тензор `(N_MU, N_L, N_B, N_M)`

### Алгоритм (broadcast)
```python
# tree: (N_M,) → (1, 1, 1, N_M)
# dU: (N_B, N_M) → (1, 1, N_B, N_M)
# Omega_L: (N_L, N_B, N_M) → (1, N_L, N_B, N_M)
# Omega_muL: (N_MU, N_L, N_B, N_M)

Omega_ren = (tree.view(1, 1, 1, N_M) 
             + dU.view(1, 1, N_B, N_M) 
             + Omega_L.view(1, N_L, N_B, N_M) 
             + Omega_muL)
```

### Сохранение кэша
```python
torch.save(dU, 'cache/dU.pt')
torch.save(Omega_L, 'cache/Omega_L.pt')
torch.save(Omega_muL, 'cache/Omega_muL.pt')
```

**Плюс:** при смене `g` (tree меняется) остальное не пересчитывается.

---

## Функция 6: Поиск минимума

### Шаг 4: Грубый минимум (argmin на сетке)

```python
# Omega_ren: (N_MU, N_L, N_B, N_M)
# Для каждой (mu, L) ищем argmin по (b, M)

Omega_2D = Omega_ren.view(N_MU, N_L, N_B * N_M)
idx_flat = torch.argmin(Omega_2D, dim=2)  # (N_MU, N_L)

idx_b = idx_flat // N_M
idx_M = idx_flat % N_M

M_coarse = M[idx_M]  # (N_MU, N_L)
b_coarse = b[idx_b]  # (N_MU, N_L)
```

### Шаг 5: Уточнение сплайном

Для каждой пары `(mu, L)`:
```
1. Взять окно 5×5 вокруг грубого минимума
   window_M = M[idx_M-2 : idx_M+3]
   window_b = b[idx_b-2 : idx_b+3]
   window_vals = Omega_ren[mu, L, idx_b-2:idx_b+3, idx_M-2:idx_M+3]

2. Построить bicubic сплайн (scipy.interpolate.RectBivariateSpline)

3. Найти минимум сплайна методом оптимизации:
   L-BFGS-B из точки (b_coarse, M_coarse)
   bounds = [window_b[0], window_b[-1]], [window_M[0], window_M[-1]]
```

### Псевдокод
```python
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

def refine_minimum(Omega_ren, M_coarse, b_coarse, M_grid, b_grid):
    M_fine = M_coarse.clone()
    b_fine = b_coarse.clone()
    
    for i in range(N_MU):
        for j in range(N_L):
            iM = torch.argmin(torch.abs(M_grid - M_coarse[i, j])).item()
            ib = torch.argmin(torch.abs(b_grid - b_coarse[i, j])).item()
            
            # Окно 5×5 с границами
            iM0, iM1 = max(0, iM-2), min(N_M, iM+3)
            ib0, ib1 = max(0, ib-2), min(N_B, ib+3)
            
            window_M = M_grid[iM0:iM1].cpu().numpy()
            window_b = b_grid[ib0:ib1].cpu().numpy()
            window_vals = Omega_ren[i, j, ib0:ib1, iM0:iM1].cpu().numpy()
            
            if window_vals.size < 4:
                continue  # окно слишком малое
            
            spline = RectBivariateSpline(window_b, window_M, window_vals)
            
            result = minimize(
                lambda x: spline.ev(x[0], x[1]),
                [b_coarse[i, j].item(), M_coarse[i, j].item()],
                bounds=[(window_b[0], window_b[-1]), (window_M[0], window_M[-1])]
            )
            
            b_fine[i, j] = result.x[0]
            M_fine[i, j] = result.x[1]
    
    return M_fine, b_fine
```

---

## Выходные данные и формат для веб-интерфейса

> ✅ **Решено:** GPU сохраняет только фазовую диаграмму (маленький файл).
> Срезы Ω(b, M) для интерактивного исследования считаются в браузере на JS —
> это быстро (< 1 сек), существующий код в `web/js/njl.js` уже это умеет.

### Что сохраняет GPU-расчёт

```python
# Финальный результат — только положения минимумов
{
  "mu":   [mu_0, mu_1, ..., mu_N],        # сетка по mu
  "L":    [L_0, L_1, ..., L_N],           # сетка по L
  "M_min": [[...], ...],                  # M в минимуме, shape (N_MU, N_L)
  "b_min": [[...], ...],                  # b в минимуме, shape (N_MU, N_L)
  "Omega_min": [[...], ...],              # значение минимума, shape (N_MU, N_L)
  "meta": {
    "g": -1.0,
    "N_b": 200, "N_M": 200,              # параметры точности расчёта
    "N_p": 100,
    "dtype": "float32",
    "computed_at": "2026-04-27"
  }
}
```

Формат файла: `web/data/phase_diagram.json`
Размер: при 100×100 сетке по (mu, L) — **~160 KB**, при 200×200 — ~640 KB.

### Что делает веб-интерфейс

```
Загрузка phase_diagram.json при старте (~160 KB, мгновенно)
         │
         ▼
Тепловая карта M_min(mu, L) и b_min(mu, L)   ← фазовая диаграмма
         │
         ▼  пользователь кликает на точку (mu_i, L_j)
         │
Вычислить Omega_ren(b, M) в JS для выбранной точки   ← быстро, < 1 сек
         │
         ▼
Срезы, компоненты, heatmap по (b, M)
```

---

## Бенчмарки и тесты

### 1. Точность float32 vs float64

```python
# Сравнить Omega_ren для одной контрольной точки
# float32 результат vs float64 результат
# Допустимая погрешность: |delta| < 1e-4 (для классификации) или < 1e-6 (для значений)
```

### 2. Сравнение с CPU-версией

```python
# Взять 10 случайных точек (mu, L)
# Посчитать M_min, b_min CPU-версией (DE) и GPU-версией
# Сравнить результаты
```

### 3. Скорость

```python
# Замерить время каждого шага:
# - dU: CPU vs GPU
# - Omega_L: CPU vs GPU
# - Omega_muL: CPU vs GPU
# - Общее время
```

### 4. Память

```python
# torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
# Убедиться что не превышаем 15 GB
```

---

## Зависимости

```
torch>=2.0          # PyTorch с CUDA
numpy               # для совместимости
scipy               # для сплайнов (RectBivariateSpline, minimize)
tqdm                # прогресс-бар
matplotlib          # визуализация
```

---

## Порядок реализации

| Этап | Что делать | Время оценочное |
|------|-----------|----------------|
| **1** | Скелет: сетки, dtype, device, конфиг | 30 мин |
| **2** | `compute_dU` на PyTorch + тест | 1–2 часа |
| **3** | `compute_Omega_L` на PyTorch + тест | 1–2 часа |
| **4** | `compute_Omega_muL` на PyTorch (сложное!) + тест | 3–5 часов |
| **5** | Сборка `Omega_ren`, broadcast, кэширование | 1 час |
| **6** | Argmin + сплайн-уточнение | 1–2 часа |
| **7** | Бенчмарки: точность, скорость, float32 vs float64 | 2–3 часа |
| **8** | Интеграция: скрипт `40_pytorch_vectorized.py` | 1 час |
| **Итого** | | **~12–18 часов** |

---

## Файлы, которые будут созданы

```
notebooks/
└── 40_pytorch_vectorized.py      # Основной скрипт (все шаги)

functions/
├── dU_torch.py                   # compute_dU на PyTorch
├── Omega_L_torch.py              # compute_Omega_L на PyTorch
├── Omega_mu_L_torch.py           # compute_Omega_muL на PyTorch (сложное)
└── minimizer_torch.py            # argmin + сплайн

cache/                            # Промежуточные результаты
├── dU.pt
├── Omega_L.pt
└── Omega_muL.pt

notebooks/output/                 # Финальные результаты
├── 40_phase_map.png
└── 40_Mb_maps.png
```

---

## Риски и запасные планы

| Риск | Запасной план |
|------|--------------|
| Omega_muL слишком медленно на GPU | Разбить на ещё меньшие батчи (~100K точек) |
| float32 даёт большую погрешность | Переключиться на float64 (время ×2, память ×2) |
| GPU память заканчивается | Батчи меньше, или `torch.cuda.empty_cache()` между шагами |
| Сплайн нестабилен на границах | Увеличить окно с 5×5 до 7×7 или 9×9 |

---

## Публикация и регистрация программы

### Регистрация в Роспатент

> ✅ **Решено:** регистрируем программу как объект интеллектуальной собственности.
> Даёт официальный номер свидетельства, на который можно ссылаться в статье.

Что регистрируется как единая программа:
- GPU-расчёт фазовой диаграммы (Python/PyTorch)
- Веб-интерфейс визуализации (HTML/JS)

Примерный порядок действий:
1. Оформить заявку на сайте ФИПС (fips.ru) — онлайн
2. Приложить: реферат (~1 страница), исходный код (выборочно), сведения об авторах
3. Срок регистрации: ~2 месяца
4. Стоимость: ~4 500 руб. для физ. лица / ~5 500 для орг.

### Статья

Тематика: **численные методы + программный инструмент** для расчёта фазовых диаграмм модели NJL3.

Ключевые результаты для статьи, которые нужно подготовить:
- Описание GPU-алгоритма (двухуровневый батчинг, ReLU-подход)
- Таблица производительности: CPU Numba vs GPU PyTorch (время, точность)
- Фазовая диаграмма высокого разрешения как иллюстрация
- Ссылка на свидетельство Роспатент + веб-интерфейс (URL)

Возможные журналы (обсудить позже):
- Computer Physics Communications
- SoftwareX
- Вестник РУДН, серия «Математика, информатика, физика»

---

## Открытые вопросы (нужно обсудить)

### ❓ Стратегия батчинга для `Omega_L`

Проблема: наивный `meshgrid(L, b, M, u1, u3)` → 50×50×50×100×100 = **1.25B float32 ≈ 5 GB** за один вызов, и таких вызовов три (для M, M=0, b=0).

Варианты:
- **A)** Батчить по `(L, b)` — внешний цикл, каждый шаг считает `(N_M, N_p, N_p)`
- **B)** Батчить по строкам `b` — как сделано в `Omega_ren_lut.py` (CPU-вариант)
- **C)** Уменьшить `N_p` для `Omega_L` (она сходится быстрее чем `Omega_mu_L`)

> Нужно определить: какой размер батча оптимален под конкретную GPU (15 GB VRAM)?

---

### ❓ Верхняя граница `p_max` для интегрирования в `Omega_mu_L`

В псевдокоде `torch.linspace(0, p_max, N_p)` — значение `p_max` не определено.

Рабочая гипотеза: `p_max = mu_max` (т.к. при `p > mu_max` всегда `E1 = sqrt(M² + p²) > mu`
и `F_± ≤ 0`, значит вклад нулевой через ReLU).

Вопросы:
- Достаточно ли `p_max = mu_max` или нужен запас?
- Использовать один глобальный `p_max` на весь батч, или адаптивный на батч?
- Как влияет выбор `p_max` на точность при малых μ (где Ферми-поверхность маленькая)?

---

*Файл обновляется по мере реализации.*

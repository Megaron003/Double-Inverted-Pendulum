# Validação de HN-B — A Não-Linearidade é Trigonométrica?

## Pêndulo Invertido Duplo

---

## 1. Processo Matemático

### 1.1 Contexto — o que HN-B acrescenta a H2

H3 já confirmou que a dinâmica não é linear (RESET potência 3, F = 724,
p ≈ 0). HN-B vai além: qual é a **classe funcional** da não-linearidade?
A resposta determina diretamente qual função de ativação a rede deve usar.

A distinção central é entre dois tipos de estrutura funcional:

**Trigonométrica / ímpar:** sin(θ), cos(θ), θ·ω — funções de ordem
ímpar em θ centrado em zero. Expandidas em série de Taylor:

$$
\sin(\theta) = \theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots
\quad \text{(apenas potências ímpares)}
$$

**Polinomial / par:** θ², ω², θ·θⱼ — funções de ordem par. Expandidas:

$$
\theta^2, \;\theta^4, \;\theta^2\omega^2, \;\ldots
\quad \text{(apenas potências pares)}
$$

A distinção par/ímpar é a chave porque a função de ativação tanh tem
exatamente a expansão de Taylor da forma ímpar:

$$
\tanh(z) = z - \frac{z^3}{3} + \frac{2z^5}{15} - \cdots
\quad \text{(apenas potências ímpares)}
$$

Enquanto ReLU gera representações lineares por partes — úteis para
aproximar funções par, mas incapazes de representar sin(θ) exatamente.

### 1.2 Hipóteses formais

$$
H_0:\; \text{a estrutura omitida pelo modelo linear é polinomial (par em } \theta\text{)}
$$

$$
\quad \Rightarrow \text{ RESET potência 2 rejeita } H_0 \text{, potência 3 não acrescenta}
$$

$$
H_1:\; \text{a estrutura omitida é trigonométrica/ímpar } (\sin\theta,\, \cos\theta,\, \theta\cdot\omega)
$$

$$
\quad \Rightarrow \text{ RESET potência 2 NÃO rejeita, potência 3 rejeita}
$$

### 1.3 Três testes encadeados

**Teste 1 — RESET potência 2 vs 3**

O RESET de Ramsey aumenta o modelo com potências dos valores ajustados:

$$
M_{\text{p}=2}: \quad \alpha_i = f(\mathbf{x}) + \gamma_2 \hat\alpha^2 + \varepsilon
$$

$$
M_{\text{p}=3}: \quad \alpha_i = f(\mathbf{x}) + \gamma_2 \hat\alpha^2 + \gamma_3 \hat\alpha^3 + \varepsilon
$$

A estatística F testa $H_0: \gamma_k = 0$ para cada potência:

$$
F = \frac{(RSS_0 - RSS_{\text{aug}})/q}{RSS_{\text{aug}}/(n-k-q)} \sim F(q,\, n-k-q)
$$

$\hat\alpha^2$ é um termo **par** — ortogonal à estrutura sin(θ) (ímpar).
$\hat\alpha^3$ é um termo **ímpar** — captura a estrutura trigonométrica.

Se $p=2$ não rejeita mas $p=3$ rejeita: assinatura diagnóstica de
**não-linearidade ímpar** — exatamente trigonométrica.

**Teste 2 — Regressão auxiliar direta**

Regride-se os resíduos do modelo linear diretamente sobre dois conjuntos
de termos candidatos, medindo $R^2_{\text{aux}}$ de cada um:

$$
r_i = \delta_0 + \delta_1\sin\theta_1 + \delta_2\cos\theta_1 + \delta_3\sin\theta_2 +
\delta_4\cos\theta_2 + \delta_5(\theta_1\omega_1) + \delta_6(\theta_2\omega_2) + \varepsilon
$$

$$
r_i = \eta_0 + \eta_1\theta_1^2 + \eta_2\theta_2^2 + \eta_3\omega_1^2 +
\eta_4\omega_2^2 + \eta_5(\theta_1\theta_2) + \eta_6(\omega_1\omega_2) + \varepsilon
$$

**Teste 3 — F-parcial aninhado**

Combina os dois conjuntos em um modelo com todos os 12 termos e testa
a contribuição incremental de cada conjunto dado o outro:

$$
H_0^{(A)}:\; \beta_{\sin,\cos,\theta\omega} = 0 \;\text{ dado poly} \quad
\rightarrow F(trig \mid poly) \sim F(6,\, n-13)
$$

$$
H_0^{(B)}:\; \beta_{\theta^2,\omega^2,...} = 0 \;\text{ dado trig} \quad
\rightarrow F(poly \mid trig) \sim F(6,\, n-13)
$$

---

## 2. Resultados

### 2.1 Teste 1 — RESET potência 2 vs 3

| Variável | RESET p=2  | p-valor | Decisão (α=0,05)   | RESET p=3   | p-valor | Decisão   |
| --------- | ---------- | ------- | -------------------- | ----------- | ------- | ---------- |
| α₁      | F =   1,15 | 0,2843  | Não rejeita H₀  ✗ | F = 724,14  | ≈ 0    | Rejeita ✓ |
| α₂      | F =   2,43 | 0,1188  | Não rejeita H₀  ✗ | F = 4131,15 | ≈ 0    | Rejeita ✓ |

**Assinatura diagnóstica:** p=2 não rejeita + p=3 rejeita = não-linearidade de natureza ímpar.

### 2.2 Teste 2 — R² da regressão auxiliar

| Conjunto de termos                  | R²_aux (α₁) | R²_aux (α₂) |
| ----------------------------------- | -------------- | -------------- |
| Trigonométrico (sin/cos/θω)      | 0,0900         | 0,1029         |
| Polinomial (θ², ω², θθ, ωω) | 0,0021         | 0,0028         |
| Combinado (trig + poly)             | 0,0929         | 0,1072         |
| Razão trig / poly                  | 42×           | 37×           |

Os termos trigonométricos explicam **42× mais** variância dos resíduos do
que os termos polinomiais em α₁, e **37×** em α₂.

### 2.3 Teste 3 — F-parcial aninhado

| Hipótese nula                        | F (α₁) | p-valor | F (α₂) | p-valor |
| ------------------------------------- | -------- | ------- | -------- | ------- |
| H₀: trig = 0 dado poly — F(6, n-13) | 834,10   | ≈ 0    | 974,34   | ≈ 0    |
| H₀: poly = 0 dado trig — F(6, n-13) | 27,14    | ≈ 0    | 39,49    | ≈ 0    |

Ambos os conjuntos acrescentam além do outro — mas a contribuição
trigonométrica é **31× maior** que a polinomial em α₁ (834 vs 27).

### 2.4 Coeficientes mais significativos (regressão auxiliar)

#### α₁

| Feature    | β       | p-valor |
| ---------- | -------- | ------- |
| sin θ₁   | +9,1436  | ≈ 0    |
| sin θ₂   | +8,2276  | ≈ 0    |
| θ₁·ω₁ | +0,2077  | 1,5e-56 |
| cos θ₂   | −0,7154 | 3,4e-04 |
| θ₂·ω₂ | +0,1173  | 8,2e-22 |

#### α₂

| Feature    | β        | p-valor |
| ---------- | --------- | ------- |
| sin θ₂   | −30,7586 | ≈ 0    |
| sin θ₁   | −7,3630  | 1,1e-48 |
| cos θ₁   | +5,7106   | 1,4e-22 |
| θ₁·ω₁ | −0,4086  | 7,0e-38 |
| cos θ₂   | +2,0178   | 3,1e-05 |
| θ₂·ω₂ | +0,1489   | 5,0e-07 |

---

## 3. Interpretação dos Resultados

### 3.1 Por que p=2 não rejeita e p=3 rejeita

Este resultado não é uma contradição — é a assinatura matemática esperada
para não-linearidade trigonométrica.

sin(θ) centrado em zero é uma função **ímpar**: sin(−θ) = −sin(θ). Isso
significa que sin(θ) é ortogonal a qualquer função par — em particular,
$\hat\alpha^2$ (par) e sin(θ) têm produto interno aproximadamente zero
em distribuições simétricas.

$$
\int_{-\pi}^{\pi} \sin(\theta) \cdot \theta^2 \, d\theta = 0
$$

O RESET com $\hat\alpha^2$ é cego à estrutura senoidal por essa ortogonalidade.
O RESET com $\hat\alpha^3$ (ímpar) captura a estrutura e rejeita com F = 724.

O padrão "p=2 não rejeita, p=3 rejeita" é por si só evidência de que a
não-linearidade é de natureza ímpar — ou seja, trigonométrica.

### 3.2 O que os coeficientes dizem fisicamente

Os termos mais significativos em ambas as variáveis são sin(θ₁) e sin(θ₂).
Isso é diretamente consistente com as equações de Lagrange do pêndulo duplo,
que contêm termos do tipo sin(θ₁ − θ₂), sin(θ₁), sin(θ₂) na expressão
das acelerações generalizadas.

A regressão auxiliar identificou esses termos a partir dos resíduos, sem
qualquer conhecimento prévio do modelo físico — exatamente o princípio
da análise cega.

### 3.3 Decisão estatística consolidada

| Critério                   | α₁             | α₂             | Julgamento                |
| --------------------------- | ---------------- | ---------------- | ------------------------- |
| RESET p=2  (par)            | não rejeita     | não rejeita     | Estrutura par ausente     |
| RESET p=3  (ímpar)         | rejeita ✓       | rejeita ✓       | Estrutura ímpar presente |
| R²_aux trig / R²_aux poly | 42×             | 37×             | Trig domina amplamente    |
| F(trig\| poly)              | 834              | 974              | Trig significativo        |
| F(poly\| trig)              | 27               | 39               | Poly marginal             |
| **H₁ confirmada**    | **Sim ✓** | **Sim ✓** | Trigonométrica           |

---

## 4. Distinção entre Formulação e Validação de HN-B

### 4.1 O que cada etapa é

| Aspecto          | Formulação                               | Validação                                           |
| ---------------- | ------------------------------------------ | ----------------------------------------------------- |
| Ponto de partida | Padrão sigmoidal observado nos resíduos  | RESET p=2 vs p=3, regressão auxiliar                 |
| Pergunta         | "A estrutura parece trigonométrica?"      | "A estrutura é ímpar? Quanto explica?"              |
| Ferramenta       | Inspeção visual do gráfico de resíduos | F-parcial, R²_aux, F aninhado                        |
| Resultado        | Suspeita qualitativa                       | Confirmação com distribuição F(q, n-k) explícita |

### 4.2 O que a validação acrescenta à formulação

A formulação observou o padrão em "S" nos resíduos vs θ e sugeriu termos
trigonométricos. A validação acrescenta quatro elementos que a inspeção
visual não pode fornecer:

**Separação par/ímpar:** o RESET distingue formalmente entre não-linearidade
polinomial e trigonométrica. A inspeção visual não faz essa separação.

**Quantificação da dominância:** R²_trig / R²_poly = 42× — a inspeção
visual não diz quantas vezes a trigonométrica domina a polinomial.

**Contribuição incremental controlada:** F(trig | poly) = 834 confirma
que os termos trigonométricos acrescentam além dos polinomiais, controlando
para a presença destes. Correlação visual não controla confundimento.

**Coeficientes com magnitude e incerteza:** β(sin θ₁) = +9,14 com p ≈ 0
quantifica o efeito. A inspeção visual diz "parece sigmoidal" sem magnitude.

### 4.3 Implicação para escolha de ativação

A formulação dizia: *"a ativação tanh ou SIREN parece mais adequada."*

A validação diz: *"os resíduos do modelo linear contêm estrutura de ordem
ímpar em θ (sin θ₁ e sin θ₂ com coeficientes β ≈ 9 e p ≈ 0, explicando
R²_aux = 0,09). Termos polinomiais de ordem par explicam apenas R²_aux = 0,002
— 42× menos. Sob α = 0,05, H₁ é confirmada pelos três testes independentes.
A ativação correta é tanh ou SIREN, porque ambas possuem expansão de
Taylor com termos ímpares que aproximam sin(θ) com qualquer precisão.
ReLU não pode representar sin(θ) exatamente."*

---

## 5. Implicação Arquitetural

HN-B confirmada significa que a função $f: \mathbf{x} \mapsto \dot{\mathbf{x}}$
contém termos da forma sin(θᵢ) e θᵢ·ωᵢ como estrutura dominante.

**Ativação tanh:** captura termos ímpares via expansão de Taylor. Cada
neurônio com peso w e bias b implementa tanh(wᵀx + b), que inclui
(wᵀx)³/3 e (wᵀx)⁵/5 — suficiente para aproximar sin(θ) com rede suficientemente
larga.

**Ativação SIREN:** usa sin(ωwᵀx + b) diretamente como ativação — a
correspondência com a estrutura do sistema é exata por construção. Indicada
quando a periodicidade é confirmada a priori, como aqui.

**Por que não ReLU:** ReLU(wᵀx) = max(0, wᵀx) gera funções contínuas
lineares por partes. Pode aproximar sin(θ) arbitrariamente bem com largura
suficiente, mas converge mais lentamente e exige mais neurônios do que
tanh/SIREN para a mesma precisão em dinâmicas trigonométricas.

---

## Referências

- Ramsey, J. B. (1969). *Tests for specification errors in classical linear
  least-squares regression analysis.* JRSS-B, 31(2).
- Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., &
  Wetzstein, G. (2020). *Implicit neural representations with periodic
  activation functions (SIREN).* NeurIPS.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and
  Panel Data.* MIT Press.

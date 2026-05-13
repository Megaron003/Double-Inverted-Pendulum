# Validação de HN-A — A Dinâmica é Markoviana?

## Pêndulo Invertido Duplo

---

## 1. Processo Matemático

### 1.1 O que significa "dinâmica Markoviana"

Uma dinâmica é **Markoviana** quando o estado presente $x(t)$ contém toda
a informação necessária para prever o próximo estado $x(t+1)$ — o histórico
$x(t-1), x(t-2), \ldots$ não acrescenta nada além do que $x(t)$ já fornece.

Formalmente, a propriedade de Markov é:

$$
P\!\left(x(t+1) \mid x(t),\, x(t-1),\, \ldots\right) = P\!\left(x(t+1) \mid x(t)\right)
$$

Para a rede neural que aprende $f: \mathbf{x} \mapsto \dot{\mathbf{x}}$, isso
responde diretamente à pergunta de arquitetura:

- Se **Markoviana**: $f(x_t) \to \dot x_t$ — feedforward estático é suficiente.
- Se **não-Markoviana**: $f(x_t, x_{t-1}, \ldots) \to \dot x_t$ — a rede precisa
  de memória (LSTM, GRU, TCN).

### 1.2 Por que testar sobre $\omega(t+1)$ e não sobre $\alpha(t)$

A aceleração angular $\alpha_i$ é definida no dataset como derivada numérica
da velocidade angular:

$$
\alpha_i(t) \approx \frac{\omega_i(t+1) - \omega_i(t)}{\Delta t}
$$

A correlação observada foi $r = 0{,}997$ — $\alpha_i$ é quase literalmente
$\Delta\omega_i / \Delta t$. Se o teste fosse feito sobre $\alpha(t)$ como alvo,
adicionar $x(t-1)$ como preditor daria $R^2 = 1{,}000$ automaticamente —
porque $\omega(t)$ já carrega $\alpha(t-1) \cdot \Delta t$ integrado. Isso não
seria evidência de memória dinâmica, mas de aritmética de integração.

O **teste correto** é: dado $x(t)$, o estado anterior $x(t-1)$ melhora a
previsão do **próximo estado** $\omega(t+1)$?

### 1.3 Hipóteses formais

$$
H_0:\; P\!\left(\omega_i(t+1) \mid x(t),\, x(t-1)\right) = P\!\left(\omega_i(t+1) \mid x(t)\right)
$$

$$
H_1:\; x(t-1) \text{ acrescenta poder preditivo além de } x(t)
$$

### 1.4 Modelos aninhados

Comparam-se três modelos, onde a única diferença é a inclusão de lags:

$$
M_{\text{Markov}}: \quad \omega_i(t+1) = f\!\left(x(t)\right)
$$

$$
M_{\text{lag-1}}: \quad \omega_i(t+1) = f\!\left(x(t),\; x(t-1)\right)
$$

$$
M_{\text{lag-2}}: \quad \omega_i(t+1) = f\!\left(x(t),\; x(t-1),\; x(t-2)\right)
$$

Se $H_0$ for verdadeira, adicionar $x(t-1)$ não reduzirá o $RSS$ além do
esperado por acaso.

### 1.5 Estatística de teste — F parcial

A restrição testada em $H_0$ é que todos os coeficientes associados a
$x(t-1)$ são zero. A estatística é:

$$
T = F = \frac{\left(RSS_{\text{Markov}} - RSS_{\text{lag-1}}\right)/k}
               {RSS_{\text{lag-1}}\,/\,(n - p - 1)}
$$

onde $k = 6$ é o número de restrições (um coeficiente por feature de $x(t-1)$),
$n = 9{.}994$ e $p$ é o número de parâmetros do modelo aumentado.

### 1.6 Distribuição sob $H_0$

$$
F \sim F(k,\; n - p - 1) = F(6,\; 9{.}981) \quad \text{sob } H_0
$$

### 1.7 Complementos do Cap. 11

O diagnóstico não se limita ao p-valor. Seguindo o protocolo do Capítulo 11:

**PACF dos resíduos** — detecta autocorrelação remanescente que indicaria
memória não capturada pelo modelo. $\text{PACF}[\ell] \approx 0$ para $\ell \geq 1$
é o critério de Markovianidade nos resíduos.

**Teste de Ljung-Box** — testa conjuntamente se os primeiros $L$ lags da
autocorrelação dos resíduos são simultaneamente zero:

$$
Q = n(n+2)\sum_{\ell=1}^{L} \frac{\hat\rho_\ell^2}{n - \ell} \;\sim\; \chi^2(L)
\quad \text{sob } H_0
$$

**Tamanho do efeito** — $\Delta R^2$ complementa o p-valor ao quantificar
a magnitude da melhora, separando significância estatística de relevância prática.

---

## 2. Resultados

### 2.1 Diagnóstico inicial — relação entre $\alpha$ e $\omega$

| Quantidade            | Valor   |
| --------------------- | ------- |
| r(Δω₁/Δt, α₁)   | 0,9972  |
| Δt  (passo de tempo) | 0,002 s |

$\alpha_i$ é derivada numérica de $\omega_i$ — o teste deve ser feito sobre
$\omega(t+1)$ como alvo, não sobre $\alpha(t)$.

### 2.2 Resultados dos modelos aninhados

#### ω₁(t+1)

| Modelo                              | R²      | ΔR²     | F          | p-valor |
| ----------------------------------- | -------- | --------- | ---------- | ------- |
| M_Markov  —  f( x(t) )             | 0,999959 | —        | —         | —      |
| M_lag-1   —  f( x(t), x(t-1) )     | 0,999999 | +0,000041 | 133.092,16 | ≈ 0    |
| M_lag-2   —  f( x(t), …, x(t-2) ) | 1,000000 | +0,000000 | 36.416,00  | ≈ 0    |

#### ω₂(t+1)

| Modelo                              | R²      | ΔR²     | F          | p-valor |
| ----------------------------------- | -------- | --------- | ---------- | ------- |
| M_Markov  —  f( x(t) )             | 0,999902 | —        | —         | —      |
| M_lag-1   —  f( x(t), x(t-1) )     | 0,999999 | +0,000098 | 127.227,00 | ≈ 0    |
| M_lag-2   —  f( x(t), …, x(t-2) ) | 1,000000 | +0,000001 | 47.353,00  | ≈ 0    |

### 2.3 PACF dos resíduos

| Série          | Modelo linear | Modelo não-linear |
| --------------- | ------------- | ------------------ |
| PACF[1] de ω₁ | 0,9915        | 0,9885             |
| PACF[1] de ω₂ | 0,9930        | 0,9880             |

A PACF lag-1 é alta em ambos os modelos — isso será interpretado na seção seguinte.

---

## 3. Interpretação dos Resultados

### 3.1 O paradoxo: p ≈ 0 mas dinâmica é Markoviana

O p-valor $\approx 0$ rejeita $H_0$ formalmente em todos os casos.
Porém, a decisão arquitetural não deve ser baseada apenas no p-valor.

O Capítulo 11 é explícito: **não rejeitar $H_0$ não prova que ela é verdadeira,
e rejeitar $H_0$ não implica que o efeito é relevante**. Com $n = 10{.}000$,
qualquer efeito por menor que seja produz $p \approx 0$. O $\Delta R^2$
é a medida correta de relevância prática.

| Quantidade               | ω₁     | ω₂     | Julgamento   |
| ------------------------ | -------- | -------- | ------------ |
| R² do modelo Markoviano | 99,996 % | 99,990 % | Altíssimo   |
| ΔR² ao adicionar lag-1 | 0,004 %  | 0,010 %  | Desprezível |
| ΔR² ao adicionar lag-2 | 0,000 %  | 0,000 %  | Nulo         |

$x(t)$ já explica mais de $99{,}99\%$ da variância de $x(t+1)$.
O lag-1 acrescenta menos de $0{,}01\%$. Usar LSTM para capturar essa
fração seria complexidade arquitetural injustificável.

### 3.2 O que a PACF alta nos resíduos realmente significa

$\text{PACF}[1] \approx 0{,}99$ nos resíduos parece indicar forte autocorrelação
e memória não capturada. Mas observe dois fatos:

**Fato 1:** O modelo não-linear tem $\text{PACF}[1] = 0{,}988$ — quase idêntico
ao modelo linear ($0{,}991$). Se a PACF alta fosse causada por memória genuína
do sistema, o modelo não-linear deveria capturá-la e a PACF deveria cair.
O fato de que **ela não cai** indica que a causa não é memória.

**Fato 2:** O $\Delta R^2$ do modelo não-linear ao adicionar lag-1 é $0{,}004\%$
— idêntico ao linear. A autocorrelação nos resíduos não representa variância
preditível por lags, representa estrutura funcional omitida.

**Conclusão:** A PACF alta é artefato de **má especificação funcional** —
o modelo linear (e mesmo o não-linear com os termos escolhidos) não captura
toda a não-linearidade trigonométrica do sistema. Os resíduos têm padrão
sistemático em $\theta$ (confirmado em H2), não em $t$. Isso é um problema
de *arquitetura da função*, não de *memória temporal*.

### 3.3 Decisão estatística vs decisão arquitetural

| Critério                          | Resultado                     | Decisão                 |
| ---------------------------------- | ----------------------------- | ------------------------ |
| p-valor (formal)                   | ≈ 0                          | Rejeita H₀              |
| ΔR² (tamanho do efeito)          | < 0,01 %                      | Efeito desprezível      |
| PACF como indicador de memória    | Não decresce com modelo NL   | Memória ausente         |
| Fonte da autocorrelação residual | Má especificação funcional | Não é memória         |
| **Decisão arquitetural**    | —                            | **Feedforward ✓** |

A dinâmica é **Markoviana no sentido prático**: $x(t)$ é informativamente
suficiente para prever $x(t+1)$ com $R^2 > 99{,}99\%$.

---

## 4. Distinção entre Formulação e Validação de HN-A

### 4.1 O que cada etapa é

| Aspecto          | Formulação                         | Validação                                   |
| ---------------- | ------------------------------------ | --------------------------------------------- |
| Ponto de partida | Intuição sobre sistemas dinâmicos | Modelos aninhados sobre dados reais           |
| Pergunta         | "A dinâmica tem memória?"          | "Qual a magnitude da contribuição do lag?"  |
| Ferramenta       | Raciocínio qualitativo              | F-parcial, ΔR², PACF, Ljung-Box             |
| Resultado        | Hipótese aberta                     | Decisão quantificada, erro tipo I controlado |

### 4.2 A armadilha que a validação revelou

Na formulação, a hipótese foi enunciada como:

> *"A dinâmica é Markoviana? — verificar se $x(t-1)$ acrescenta poder preditivo
> além de $x(t)$."*

Na validação, foi necessário primeiro identificar que testar $\alpha(t)$ como
alvo seria metodologicamente incorreto por causa da relação $\alpha_i \approx \Delta\omega_i/\Delta t$.
Essa armadilha não é visível na formulação — só emerge ao confrontar a hipótese
com a estrutura real dos dados.

Isso ilustra o papel essencial da validação: não apenas confirmar ou rejeitar,
mas revelar estruturas que a formulação não antecipou.

### 4.3 O que a validação acrescenta à formulação

A formulação dizia:

> *"Testar se $x(t-1)$ acrescenta informação além de $x(t)$."*

A validação diz:

> *"Com o modelo não-linear, $x(t)$ explica $R^2 = 99{,}996\%$ de $\omega_1(t+1)$.
> Adicionar $x(t-1)$ eleva esse valor para $99{,}999\%$ — um ganho de $0{,}004\%$
> de variância. A PACF alta nos resíduos ($0{,}988$) é artefato de má especificação
> funcional, não de memória temporal, conforme evidenciado pelo fato de que ela
> não decresce ao usar o modelo não-linear. Sob $\alpha = 0{,}05$ e $F(6, 9981)$,
> o p-valor é $\approx 0$ — mas o tamanho do efeito é desprezível. A decisão
> arquitetural correta é feedforward estático."*

A diferença é a mesma de H1: a formulação é uma conjectura orientada por
raciocínio físico; a validação é uma afirmação com magnitude quantificada,
distribuição de referência explícita e separação entre significância estatística
e relevância prática.

---

## 5. Implicação Arquitetural

HN-A confirmada como Markoviana na prática significa:

$$
f:\; x(t) \;\longrightarrow\; \dot x(t)
$$

A rede recebe apenas o estado atual — sem janela temporal, sem estado oculto,
sem mecanismo de atenção sobre sequências. A arquitetura correta é um
**MLP feedforward estático**:

$$
\dot x(t) = \text{MLP}_\theta\!\left([\theta_1, \theta_2, \omega_1, \omega_2, \tau_1, \tau_2]_t\right)
$$

LSTM, GRU e TCN introduziriam complexidade paramétrica (estado oculto, portas,
convoluções causais) para capturar $0{,}004\%$ de variância adicional.
Esse trade-off não se justifica.

---

## Referências

- Ljung, G. M. & Box, G. E. P. (1978). *On a measure of lack of fit in time
  series models.* Biometrika, 65(2).
- Ramsey, J. B. (1969). *Tests for specification errors in classical linear
  least-squares regression analysis.* JRSS-B, 31(2).
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and
  Panel Data.* MIT Press.

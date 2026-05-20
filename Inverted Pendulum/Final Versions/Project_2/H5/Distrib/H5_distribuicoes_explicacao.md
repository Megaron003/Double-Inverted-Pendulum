# Distribuição de H₀ — HN-F
## Como ocorre a probabilidade de ocorrência dado o dataset do simulador

---

## 1. O Problema da População

Os dados foram extraídos de um simulador de física hiper-realista (MuJoCo),
não de uma população real de pêndulos físicos. Isso levanta uma questão
metodológica fundamental:

> Se não há população, como se constrói a distribuição de H₀?

A resposta está na distinção entre dois tipos de distribuição:

**Distribuição dos dados** — como os valores de $\alpha_1$, $\theta_2$, $\omega_1$
se distribuem no dataset. Essa distribuição depende do simulador, das condições
iniciais e do controlador. Ela é empírica e específica ao experimento.

**Distribuição amostral da estatística** — como a estatística $F$ se distribuiria
em repetições hipotéticas do experimento, assumindo que $H_0$ é verdadeira.
Essa distribuição é **analítica** — derivada matematicamente a partir dos
pressupostos do modelo de regressão, independente da origem dos dados.

O que é plotado nos gráficos de probabilidade de ocorrência é a segunda:
a distribuição amostral de $F$ sob $H_0$.

---

## 2. O Que é a Distribuição Amostral de F

### 2.1 Construção formal

Sob $H_0$ (o componente testado é irrelevante), a estatística F é construída
como razão de duas estimativas independentes da mesma variância:

$$F = \frac{(RSS_{\text{restrito}} - RSS_{\text{completo}})/q}{RSS_{\text{completo}}/(n-k)}$$

O numerador mede a melhora de ajuste ao adicionar o componente testado,
normalizada pelo número de restrições $q$. O denominador é a variância
residual média do modelo completo.

Gauss, Fisher e Snedecor demonstraram que, quando $H_0$ é verdadeira e
os erros são independentes com variância constante, o numerador e o denominador
são estimativas independentes de $\sigma^2$. A razão de duas somas de
qui-quadrado independentes, divididas por seus graus de liberdade, segue
exatamente:

$$F \sim F(q,\; n-k) \quad \text{sob } H_0$$

Essa não é uma aproximação — é uma distribuição exata derivada analiticamente.

### 2.2 O que "probabilidade de ocorrência" significa aqui

A PDF da distribuição $F(df_1, df_2)$ responde à pergunta:

> Se o simulador gerasse infinitas amostras de $n = 50.000$ pontos com
> $\sin\theta_2$ genuinamente irrelevante para $\alpha_1$, com que frequência
> observaríamos cada valor de $F$?

O eixo $y$ dos gráficos é essa frequência relativa — a densidade de
probabilidade de ocorrência de cada valor de $F$ sob $H_0$.

### 2.3 Por que funciona sem população real

O raciocínio não requer uma população física de pêndulos. Ele requer apenas:

1. Um modelo de regressão linear bem especificado
2. Erros aproximadamente independentes
3. Variância residual aproximadamente constante

Esses pressupostos foram verificados via diagnósticos do Capítulo 11
(resíduos vs ajustados, Q-Q, Cook, VIF). Dado que os pressupostos são
satisfeitos, a distribuição $F(df_1, df_2)$ é a distribuição correta da
estatística sob $H_0$ — independente de os dados virem de um simulador,
de um experimento físico ou de qualquer outra fonte.

---

## 3. Os Três Testes de HN-F e Suas Distribuições

### 3.1 Teste 1 — sin θ₂ dado cos θ₂

$$H_0:\; \beta_{\sin\theta_2} = 0 \quad \text{dado que } \cos\theta_2 \text{ está presente}$$

$$T = F = \frac{(RSS_{M_{\text{sem\_sin}}} - RSS_{M_{\text{full}}})/1}{RSS_{M_{\text{full}}}/(n-6)} \sim F(1,\; 49994) \quad \text{sob } H_0$$

| Parâmetro | Valor |
|-----------|-------|
| Distribuição sob H₀ | F(1, 49994) |
| F_crítico (α=0,05) | 3,842 |
| T_obs | 25.137,52 |
| p-valor | ≈ 0 |
| ΔR² | 26,491% |

A distribuição F(1, 49994) é extremamente concentrada próxima de 1 — com
$df_2 = 49994$, o denominador é estimado com altíssima precisão e a
variância da distribuição é aproximadamente $2/(df_2-2) \approx 0,00004$.

T_obs = 25.137 está a milhares de desvios-padrão da média da distribuição
sob $H_0$. A seta sai do gráfico porque o eixo teria que ir até 25.137
para mostrar onde T_obs cai — e a curva já teria desaparecido visualmente
muito antes disso.

**O efeito é real:** ΔR² = 26,5% significa que sin θ₂ explica 26,5 pontos
percentuais adicionais de variância de α₁. Esse F enorme não é artefato
do tamanho amostral — é consequência de um efeito genuinamente grande.

### 3.2 Teste 2 — cos θ₂ dado sin θ₂

$$H_0':\; \beta_{\cos\theta_2} = 0 \quad \text{dado que } \sin\theta_2 \text{ está presente}$$

$$T = F = \frac{(RSS_{M_{\text{sem\_cos}}} - RSS_{M_{\text{full}}})/1}{RSS_{M_{\text{full}}}/(n-6)} \sim F(1,\; 49994) \quad \text{sob } H_0'}$$

| Parâmetro | Valor |
|-----------|-------|
| Distribuição sob H₀ | F(1, 49994) |
| F_crítico (α=0,05) | 3,842 |
| T_obs | 3,965 |
| p-valor | 0,0465 |
| ΔR² | 0,004% |

Aqui T_obs = 3,965 cai **dentro** do gráfico — marginalmente acima de
F_crit = 3,842. A diferença é visualmente quase imperceptível na curva,
o que revela o ponto central deste teste:

O p-valor = 0,0465 é menor que α = 0,05, portanto formalmente rejeita H₀.
Mas ΔR² = 0,004% significa que cos θ₂ explica quatro milésimos de ponto
percentual de variância adicional.

**Esse é o paradoxo do tamanho amostral em ação:**

$$F \approx \frac{\Delta R^2 \times n}{q \times (1 - R^2_{\text{completo}})}
= \frac{0{,}00004178 \times 50000}{1 \times (1 - 0{,}473)} = 3{,}97$$

Com n = 1.000, o mesmo ΔR² produziria F ≈ 0,08 — invisível. A rejeição
de H₀ é real no sentido estatístico, mas o efeito é desprezível no
sentido prático. A distribuição F(1, 49994) é tão estreita que qualquer
efeito, por menor que seja, produz F acima do crítico.

### 3.3 Teste 3 — conjunto (sin + cos θ₂)

$$H_0^{(3)}:\; \beta_{\sin\theta_2} = \beta_{\cos\theta_2} = 0$$

$$
T = F = \frac{(RSS_{M_{\text{sem\_tudo}}} - RSS_{M_{\text{full}}})/2}{RSS_{M_{\text{full}}}/(n-6)} \sim F(2,\; 49994) \quad \text{sob } H_0^{(3)}$$

| Parâmetro | Valor |
|-----------|-------|
| Distribuição sob H₀ | F(2, 49994) |
| F_crítico (α=0,05) | 2,996 |
| T_obs | 12.573,06 |
| p-valor | ≈ 0 |
| ΔR² | 26,500% |

Este é o teste original de H2. A distribuição F(2, 49994) tem dois graus
de liberdade no numerador porque duas restrições são testadas simultaneamente.
T_obs = 12.573 é dominado pela contribuição de sin θ₂ — cos θ₂ contribui
com apenas 0,02% do ganho conjunto.

---

## 4. A Distribuição Sob H₁ — F Não-Central

Quando H₁ é verdadeira (o componente testado realmente contribui), a
mesma estatística F não segue mais a distribuição central F(df₁, df₂).
Ela segue uma **distribuição F não-central** com parâmetro de
não-centralidade λ:

$$\lambda = F_{\text{obs}} \times q$$

O parâmetro λ mede o quanto o componente testado realmente explica, em
unidades de variância residual. Quanto maior λ, mais deslocada para a
direita fica a distribuição sob H₁.

| Teste | λ = F_obs × q | Centro sob H₁ | Centro sob H₀ |
|-------|---------------|---------------|---------------|
| sin θ₂ | 25.137 × 1 = 25.137 | ≈ 25.137 | ≈ 1 |
| cos θ₂ | 3,96 × 1 = 3,96 | ≈ 3,96 | ≈ 1 |
| conjunto | 12.573 × 2 = 25.146 | ≈ 12.573 | ≈ 1 |

Para sin θ₂ e o conjunto, as distribuições sob H₀ e H₁ estão completamente
separadas — não há sobreposição. O poder do teste é 100%.

Para cos θ₂, a separação é marginal: λ = 3,96 significa que H₁ está
centrada em ≈ 3,96, enquanto H₀ está centrada em ≈ 1. As distribuições
se sobrepõem substancialmente — o poder é de apenas 45%.

---

## 5. A Curva p-valor vs ΔR²

O painel inferior da figura plota o p-valor como função de ΔR² para
diferentes tamanhos de amostra. Ele ilustra a relação:

$$F \approx \frac{\Delta R^2 \times n}{q \times (1 - R^2_{\text{completo}})}$$

Para α = 0,05 e F_crit ≈ 4 (df₁ = 1, df₂ grande), o ΔR² mínimo
detectável como significativo é:

$$\Delta R^2_{\text{mín}} \approx \frac{4 \times q \times (1-R^2)}{n}
= \frac{4 \times 1 \times 0{,}527}{n}$$

| n | ΔR² mínimo detectável |
|---|----------------------|
| 1.000 | 0,211% |
| 10.000 | 0,021% |
| 50.000 | 0,004% |
| 100.000 | 0,002% |

Com n = 50.000, o limiar de detecção é ΔR² = 0,004% — exatamente onde
cos θ₂ se encontra (ΔR² = 0,004178%). Isso explica por que cos θ₂ está
no limiar da significância: com metade dos dados, não seria detectado.

---

## 6. Interpretação Consolidada

| | sin θ₂ | cos θ₂ |
|--|--------|--------|
| T_obs | 25.137 | 3,96 |
| Posição na curva F | Fora do gráfico | No limiar |
| p-valor | ≈ 0 | 0,046 |
| ΔR² | 26,49% | 0,004% |
| Origem do F grande | Efeito real | n grande |
| Poder do teste | 100% | 45% |
| Implicação | Feature essencial | Feature dispensável |
| Decisão arquitetural | Incluir sin θ₂ ✓ | cos θ₂ opcional |

A probabilidade de ocorrência sob H₀ serve aqui como régua de calibração:
ela mostra não apenas se T_obs cai na região de rejeição, mas **o quão
distante** T_obs está do centro da distribuição sob H₀. Para sin θ₂,
a distância é de 25.136 unidades — o efeito é tão grande que a distribuição
sob H₀ é irrelevante como referência. Para cos θ₂, T_obs está a 0,12
unidades além do limiar — no limite do que a distribuição F(1, 49994)
considera improvável.

---

## Referências

- Fisher, R. A. (1925). *Statistical Methods for Research Workers.* Oliver & Boyd.
- Snedecor, G. W. & Cochran, W. G. (1989). *Statistical Methods.* Iowa State University Press.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data.* MIT Press.

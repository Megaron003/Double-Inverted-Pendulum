# As Distribuições de H₀ e H₁ — Fundamento Matemático Completo
## Pêndulo Invertido Duplo · Testes de Hipótese HN-A a HN-F

---

## 1. A Pergunta Fundamental

Quando se calcula uma estatística de teste $T_{\text{obs}}$ a partir dos dados
e se obtém um p-valor, está-se respondendo implicitamente a uma pergunta precisa:

> **"Dado que $H_0$ é verdadeira, com que probabilidade observaríamos
> um resultado tão extremo quanto $T_{\text{obs}}$ — ou mais extremo?"**

Essa pergunta exige uma **distribuição de referência**: a distribuição que
$T$ seguiria se $H_0$ fosse verdadeira. Sem ela, não há como avaliar se
$T_{\text{obs}}$ é comum ou raro sob $H_0$.

O gráfico de probabilidade de ocorrência plota exatamente essa distribuição —
com $T_{\text{obs}}$ marcado para mostrar onde o resultado observado cai.

---

## 2. O Modelo de Regressão Linear — Ponto de Partida

Todos os testes realizados partem do mesmo modelo:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\, \sigma^2\mathbf{I})$$

onde $\mathbf{y} \in \mathbb{R}^n$ são as acelerações angulares observadas,
$\mathbf{X} \in \mathbb{R}^{n \times k}$ é a matriz de features,
$\boldsymbol{\beta} \in \mathbb{R}^k$ são os coeficientes e
$\sigma^2$ é a variância dos erros.

O estimador de mínimos quadrados ordinários (MQO) é:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

com vetor de resíduos $\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$
e soma dos quadrados dos resíduos:

$$RSS = \hat{\boldsymbol{\varepsilon}}^\top\hat{\boldsymbol{\varepsilon}}
= \mathbf{y}^\top(\mathbf{I} - \mathbf{H})\mathbf{y}$$

onde $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$
é a matriz de projeção (*hat matrix*).

---

## 3. A Construção da Estatística F

### 3.1 Comparação de modelos aninhados

Em todos os testes comparam-se dois modelos:

$$M_0 \text{ (restrito)}: \quad \mathbf{y} = \mathbf{X}_0\boldsymbol{\beta}_0 + \boldsymbol{\varepsilon}$$

$$M_1 \text{ (completo)}: \quad \mathbf{y} = \mathbf{X}_1\boldsymbol{\beta}_1 + \boldsymbol{\varepsilon}$$

onde $\mathbf{X}_0 \subset \mathbf{X}_1$ ($M_0$ é $M_1$ com $q$ colunas removidas).
A hipótese nula é $H_0: \boldsymbol{\beta}_{\text{extra}} = \mathbf{0}$ (as $q$ features
adicionais têm coeficientes todos nulos).

A estatística F mede a melhora de ajuste ao adicionar as $q$ features,
normalizada pela variabilidade residual do modelo completo:

$$T = F = \frac{(RSS_0 - RSS_1)/q}{RSS_1/(n - k_1 - 1)}$$

### 3.2 Decomposição algébrica

A quantidade $RSS_0 - RSS_1$ mede quanto a soma dos erros quadráticos
caiu ao adicionar as novas features. Algebricamente:

$$RSS_0 - RSS_1 = \mathbf{y}^\top(\mathbf{H}_1 - \mathbf{H}_0)\mathbf{y}$$

onde $\mathbf{H}_1 - \mathbf{H}_0$ é uma matriz de projeção ortogonal de posto $q$.
Isso significa que $RSS_0 - RSS_1$ mede a variância de $\mathbf{y}$ projetada
no subespaço complementar — exatamente a parte de $\mathbf{y}$ explicada pelas
novas features e não pelo modelo restrito.

### 3.3 Relação com o R²

A estatística F se relaciona com $\Delta R^2$ pela identidade:

$$F = \frac{\Delta R^2 / q}{(1 - R^2_1)/(n - k_1 - 1)}
\approx \frac{\Delta R^2 \cdot n}{q \cdot (1 - R^2_1)}$$

Esta identidade expõe o **paradoxo do tamanho amostral**:
$F$ cresce linearmente com $n$ para qualquer $\Delta R^2 > 0$ fixo.
Com $n = 50.000$, até $\Delta R^2 = 0{,}004\%$ produz $F > 3{,}8$.

---

## 4. A Distribuição de F Sob H₀ — Derivação

### 4.1 Duas somas de qui-quadrado independentes

Sob $H_0$ e assumindo $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$,
pode-se mostrar que:

$$\frac{RSS_0 - RSS_1}{\sigma^2} \sim \chi^2(q)$$

$$\frac{RSS_1}{\sigma^2} \sim \chi^2(n - k_1 - 1)$$

e que as duas quantidades são **independentes** (pelo teorema de Cochran,
pois as matrizes de projeção $\mathbf{H}_1 - \mathbf{H}_0$ e $\mathbf{I} - \mathbf{H}_1$
são ortogonais).

### 4.2 A distribuição F como razão de qui-quadrados

Por definição, a razão de dois qui-quadrados independentes divididos por
seus graus de liberdade segue a distribuição F de Fisher-Snedecor:

$$F = \frac{\chi^2(q)/q}{\chi^2(n-k_1-1)/(n-k_1-1)} \sim F(q,\; n-k_1-1)$$

Portanto, sob $H_0$:

$$\boxed{T = F \sim F(df_1,\; df_2), \quad df_1 = q, \quad df_2 = n - k_1 - 1}$$

### 4.3 A PDF da distribuição F

A função densidade de probabilidade de $F(df_1, df_2)$ é:

$$f(x;\, df_1, df_2) = \frac{1}{B(df_1/2,\, df_2/2)}
\left(\frac{df_1}{df_2}\right)^{df_1/2}
\frac{x^{df_1/2 - 1}}{\left(1 + \frac{df_1}{df_2}\,x\right)^{(df_1+df_2)/2}},
\quad x > 0$$

onde $B(\cdot,\cdot)$ é a função Beta. Esta é a **curva azul** nos gráficos —
a probabilidade de ocorrência de cada valor de $F$ se $H_0$ for verdadeira.

### 4.4 Propriedades da forma da curva

**Começa em zero:** $F$ é razão de variâncias, logo $F \geq 0$ sempre.

**Pico próximo de 1:** sob $H_0$, numerador e denominador estimam
a mesma $\sigma^2$, então $\mathbb{E}[F] = df_2/(df_2 - 2) \approx 1$
para $df_2$ grande.

**Cauda direita longa:** valores grandes de $F$ são raros mas possíveis —
a distribuição é assimétrica à direita.

**Concentração com $df_2$ grande:** $\text{Var}[F] = 2df_2^2(df_1+df_2-2) /
[df_1(df_2-2)^2(df_2-4)]$. Para $df_2 = 49994$, essa variância é
$\approx 2/df_1$ — a distribuição fica extremamente estreita, com
praticamente toda a massa entre 0,5 e 1,5.

---

## 5. O P-valor — Área da Cauda

### 5.1 Definição formal

$$p = P(F \geq T_{\text{obs}} \mid H_0) = 1 - \text{CDF}_{F(df_1, df_2)}(T_{\text{obs}})$$

$$= \int_{T_{\text{obs}}}^{\infty} f(x;\, df_1, df_2)\, dx$$

Esta é a **área sob a curva azul à direita de $T_{\text{obs}}$** — a
probabilidade de ocorrência de um resultado tão extremo quanto o observado,
ou mais extremo, se $H_0$ fosse verdadeira.

### 5.2 A região de rejeição

O valor crítico $F_{\text{crit}}$ é definido por:

$$P(F \geq F_{\text{crit}} \mid H_0) = \alpha \implies
F_{\text{crit}} = \text{CDF}_{F(df_1, df_2)}^{-1}(1 - \alpha)$$

A **área vermelha** nos gráficos é exatamente essa área $= \alpha = 0{,}05$.
Se $T_{\text{obs}} > F_{\text{crit}}$, então $p < \alpha$ e rejeita-se $H_0$.

### 5.3 O que o p-valor NÃO é

O p-valor não é:
- A probabilidade de $H_0$ ser verdadeira
- A probabilidade de o resultado ser falso
- Uma medida de tamanho do efeito

É apenas: a probabilidade de observar $T \geq T_{\text{obs}}$ **se** $H_0$
fosse verdadeira. Uma probabilidade sobre o espaço amostral, não sobre
o espaço das hipóteses.

---

## 6. A Distribuição de F Sob H₁ — A F Não-Central

### 6.1 O que muda quando H₁ é verdadeira

Quando $H_0$ é falsa, as features adicionadas têm coeficientes verdadeiros
$\boldsymbol{\beta}_{\text{extra}} \neq \mathbf{0}$. Nesse caso, o numerador
$RSS_0 - RSS_1$ não segue mais um qui-quadrado central — segue um
**qui-quadrado não-central**:

$$\frac{RSS_0 - RSS_1}{\sigma^2} \sim \chi^2(q,\; \lambda)$$

onde $\lambda$ é o **parâmetro de não-centralidade**:

$$\lambda = \frac{\boldsymbol{\beta}_{\text{extra}}^\top
(\mathbf{X}_{\text{extra}}^\top\mathbf{M}_0\mathbf{X}_{\text{extra}})
\boldsymbol{\beta}_{\text{extra}}}{\sigma^2}$$

e $\mathbf{M}_0 = \mathbf{I} - \mathbf{H}_0$ é a matriz de resíduos do modelo restrito.

### 6.2 A F não-central

A razão de um qui-quadrado não-central pelo mesmo qui-quadrado central
do denominador segue a **distribuição F não-central**:

$$F \sim F(df_1,\; df_2,\; \lambda) \quad \text{sob } H_1$$

A PDF da F não-central é:

$$f(x;\, df_1, df_2, \lambda) = \sum_{j=0}^{\infty}
\frac{e^{-\lambda/2}(\lambda/2)^j}{j!}
\cdot f_{\text{central}}(x;\, df_1 + 2j,\, df_2)$$

ou seja, uma mistura ponderada de distribuições F centrais com graus de
liberdade deslocados — cada componente ponderado pela distribuição de
Poisson com média $\lambda/2$.

### 6.3 O parâmetro de não-centralidade na prática

Na prática, estimamos $\lambda$ pelo valor observado:

$$\hat\lambda = F_{\text{obs}} \times df_1$$

Essa estimativa é motivada pelo fato de que $\mathbb{E}[F] = df_2(df_1 + \lambda) /
[df_1(df_2 - 2)]$, que para $df_2$ grande se aproxima de $(df_1 + \lambda)/df_1$.
Resolvendo: $\hat\lambda \approx F_{\text{obs}} \times df_1$.

### 6.4 Deslocamento da distribuição

A distribuição F não-central é deslocada para a direita em relação à F
central. O centro aproximado (moda) da F não-central é:

$$\text{moda} \approx \frac{df_1 + \lambda}{df_1} \cdot \frac{df_2 - 2}{df_2 + 2}
\approx \frac{\lambda}{df_1} = F_{\text{obs}}$$

Isso explica por que a **curva verde** nos gráficos H₀ vs H₁ está centrada
aproximadamente em $F_{\text{obs}}$ — ela mostra onde a distribuição de $F$
ficaria se o efeito observado fosse o efeito real do sistema.

---

## 7. O Poder do Teste

### 7.1 Definição

O poder do teste é a probabilidade de rejeitar $H_0$ quando $H_1$ é verdadeira:

$$\text{Poder} = P(F \geq F_{\text{crit}} \mid H_1)
= 1 - \text{CDF}_{F(df_1, df_2, \lambda)}(F_{\text{crit}})$$

É a área sob a curva verde à direita de $F_{\text{crit}}$.

### 7.2 Separação entre H₀ e H₁

Quando as duas distribuições são completamente separadas (como em HN-B
RESET ímpar e HN-C), o poder é praticamente 1 — a distribuição sob H₁
está tão à direita de $F_{\text{crit}}$ que a probabilidade de não rejeitar
H₀ quando H₁ é verdadeira é virtualmente zero.

Quando as distribuições se sobrepõem (como em HN-A com cos θ₂), o poder
é menor — há risco real de erro tipo II (não rejeitar H₀ quando ela
é falsa).

### 7.3 A relação entre poder, $\lambda$ e $\Delta R^2$

$$\lambda = F_{\text{obs}} \times df_1
\approx \frac{\Delta R^2 \cdot n \cdot df_1}{df_1 \cdot (1-R^2_1)}
= \frac{\Delta R^2 \cdot n}{1 - R^2_1}$$

O parâmetro de não-centralidade cresce linearmente com $n$ e com $\Delta R^2$.
Para $\Delta R^2$ fixo, dobrar $n$ dobra $\lambda$ e deslocando a F não-central
ainda mais para a direita — aumentando o poder. Para $n$ fixo, dobrar
$\Delta R^2$ tem o mesmo efeito.

---

## 8. O Que Cada Curva Nos Mostra de Verdade

### 8.1 A curva azul (F central sob H₀)

**Responde:** "Se as features testadas fossem completamente irrelevantes,
qual seria a frequência de cada valor de $F$ em amostras de tamanho $n$
deste sistema?"

Ela não descreve os dados — descreve a **estatística calculada sobre os dados**
em um mundo hipotético onde $H_0$ é verdadeira. É uma distribuição sobre o
espaço amostral, não sobre o espaço dos parâmetros.

### 8.2 A área vermelha (região de rejeição)

**Responde:** "Quais valores de $F$ seriam tão improváveis sob $H_0$ que,
se observados, levariam à rejeição de $H_0$ com taxa de erro tipo I de $\alpha$?"

A área vermelha tem área exatamente igual a $\alpha = 0{,}05$ — por construção,
$F_{\text{crit}}$ é definido para que isso seja verdade.

### 8.3 A linha pontilhada ($T_{\text{obs}}$)

**Responde:** "Onde o resultado observado nos dados cai na distribuição hipotética?"

Se cai **dentro da curva** (à esquerda de $F_{\text{crit}}$): os dados são
compatíveis com $H_0$. A probabilidade de observar algo tão extremo sob $H_0$
não é suficientemente pequena para rejeitar $H_0$.

Se cai **na área vermelha ou além** (à direita de $F_{\text{crit}}$): os dados
são incompatíveis com $H_0$ ao nível $\alpha$. O resultado seria
improvável demais se $H_0$ fosse verdadeira.

### 8.4 A curva verde (F não-central sob H₁)

**Responde:** "Se o efeito observado nos dados fosse o efeito real do sistema,
como seria a distribuição de $F$ em repetições do experimento?"

Ela mostra o mundo alternativo — onde as features testadas realmente
contribuem com a magnitude $\hat\lambda = F_{\text{obs}} \times df_1$.
A sobreposição entre as curvas azul e verde determina o poder do teste.

---

## 9. H₀ e H₁ Mediante a Quê?

Esta é a pergunta central — e a resposta é precisa.

### 9.1 Mediante à distribuição amostral da estatística

As distribuições de H₀ e H₁ são **distribuições da estatística $F$**,
não dos dados. Elas descrevem o comportamento de $F$ em repetições hipotéticas
do experimento — se o MuJoCo gerasse infinitas amostras de $n = 50.000$
pontos, cada uma produziria um valor de $F$. O conjunto de todos esses
valores seguiria a distribuição plotada.

### 9.2 Mediante às condições de cada hipótese

**Sob H₀** ($\lambda = 0$, features irrelevantes):
- Os dados variam aleatoriamente em torno do modelo restrito
- A melhora de ajuste $RSS_0 - RSS_1$ é puro ruído amostral
- $F$ segue $F(df_1, df_2)$ centrado em $\approx 1$

**Sob H₁** ($\lambda > 0$, features relevantes):
- Os dados têm estrutura real captada pelas features adicionais
- A melhora $RSS_0 - RSS_1$ reflete tanto o sinal real quanto o ruído
- $F$ segue $F(df_1, df_2, \lambda)$ deslocado para a direita

### 9.3 Sem população real — ainda assim válido

Os dados vieram do MuJoCo, não de uma população física de pêndulos.
A distribuição amostral de $F$ não requer população real — requer apenas:

1. Que o modelo de regressão linear seja uma aproximação razoável
   para a relação local entre features e target
2. Que os erros sejam aproximadamente independentes com variância constante
3. Que $n$ seja suficientemente grande para as aproximações assintóticas
   serem válidas

Os três pressupostos foram verificados via diagnósticos do Capítulo 11:
VIF (multicolinearidade), resíduos vs ajustados (homocedasticidade),
Q-Q (normalidade), Cook + leverage (influência). A distribuição $F$ é
matematicamente derivada — ela existe independente da origem dos dados.

---

## 10. Cada Hipótese e Sua Distribuição

### 10.1 HN-A — Markovianidade

$$F \sim F(12,\; 9969) \quad \text{sob } H_0$$

$df_1 = 12$ porque se testam 12 features de lag-1 (6 features × 1 lag).
$df_2 = 9969$ porque o episódio tem 10.000 pontos menos os 6 parâmetros
do modelo não-linear mais 1.

$T_{\text{obs}} = 133.092$ com $\Delta R^2 = 0{,}004\%$. A distribuição
$F(12, 9969)$ tem variância $\approx 2/9 \approx 0{,}22$ — extremamente
estreita. Qualquer $\Delta R^2 > 0$ produz $F$ enorme por causa de $n$.

**Lição:** F enorme não implica efeito grande. $\Delta R^2 = 0{,}004\%$
confirma que a rejeição é artefato do tamanho amostral.

### 10.2 HN-B — Não-linearidade (RESET)

**RESET par** $(df_1 = 1, df_2 = 49994)$: $F_{\text{obs}} = 1{,}15$ para α₁.
T_obs cai **dentro** da curva F(1, 49994). Isso não é falha — é o resultado
esperado matematicamente porque $\hat\alpha^2$ (par) é ortogonal a $\sin\theta$
(ímpar). A ortogonalidade faz o produto interno entre o espaço gerado por
$\hat\alpha^2$ e o espaço de sin(θ) tender a zero:

$$\int_{-\pi}^{\pi} \sin(\theta) \cdot \theta^2 \, d\theta = 0$$

**RESET ímpar** $(df_1 = 2, df_2 = 49993)$: $F_{\text{obs}} = 724$ para α₁.
$\hat\alpha^3$ (ímpar) tem produto interno não-nulo com $\sin\theta$ — o
RESET captura a estrutura e F dispara.

**A assimetria par/ímpar é a prova matemática de que a não-linearidade é sin.**

### 10.3 HN-C — τ necessário

$$F \sim F(2,\; 49993) \quad \text{sob } H_0$$

$df_1 = 2$ porque se testam 2 coeficientes ($\beta_{\tau_1} = \beta_{\tau_2} = 0$).
$T_{\text{obs}} = 120.304$ com $\Delta R^2 = 57{,}9\%$.

Aqui $\lambda = 120.304 \times 2 = 240.608$. A distribuição sob H₁ está
centrada em $\approx 120.304$ — separada por 120.303 unidades da
distribuição sob H₀ (centrada em 1). Poder = 100%.

**Diferença de HN-A:** ambas têm $F \approx 10^5$, mas:

| | HN-A | HN-C |
|--|------|------|
| $\Delta R^2$ | 0,004% | 57,9% |
| $\lambda$ | $133.092 \times 12 \approx 1{,}6 \times 10^6$ | $120.304 \times 2 \approx 240.608$ |
| F grande por | n = 10.000 grande | $\Delta R^2$ grande |

Os F são da mesma ordem — os $\lambda$ diferem por 6×. HN-A tem $\lambda$
maior apenas porque $df_1 = 12$ amplifica.

### 10.4 HN-D — Distribuição t de Student

$$\text{LRT} = 2[\ell_t(\hat\theta_t) - \ell_n(\hat\theta_n)] \sim \chi^2(1) \quad \text{sob } H_0$$

Aqui a distribuição não é F — é qui-quadrado com 1 grau de liberdade,
porque se testa uma única restrição ($\nu = \infty$, equivalente à Normal).

$$\text{LRT}_{\text{obs}} = 8555 \quad p = P(\chi^2(1) \geq 8555) \approx 0$$

O AIC adiciona a penalidade de parâmetros: $\Delta\text{AIC} = 8553$,
esmagadoramente a favor de $\nu < \infty$ (t de Student).

### 10.5 HN-F — sin θ₂ vs cos θ₂

**sin θ₂** $(df_1 = 1, df_2 = 49994)$: $F_{\text{obs}} = 25.137$, $\lambda = 25.137$.
A F não-central está centrada em $\approx 25.137$ — separação completa de H₀.

**cos θ₂** $(df_1 = 1, df_2 = 49994)$: $F_{\text{obs}} = 3{,}96$, $\lambda = 3{,}96$.
A F não-central está centrada em $\approx 3{,}96$ — marginalmente além de H₀
(centrada em 1). As distribuições se sobrepõem substancialmente.
Poder ≈ 45%.

**A comparação entre sin e cos θ₂ é o exemplo mais limpo do par**
**(significância estatística vs relevância prática):**

| | sin θ₂ | cos θ₂ |
|--|--------|--------|
| $F_{\text{obs}}$ | 25.137 | 3,96 |
| $\lambda$ | 25.137 | 3,96 |
| $\Delta R^2$ | 26,49% | 0,004% |
| Separação H₀/H₁ | 25.136 unidades | 2,96 unidades |
| Poder | 100% | 45% |
| Origem do F | Efeito real | n = 50.000 |

---

## 11. A Curva p-valor vs ΔR² — O Mapa do Paradoxo

A relação:

$$p \approx 1 - \text{CDF}_{F(df_1, df_2)}\!\left(\frac{\Delta R^2 \cdot n}{df_1 \cdot (1-R^2)}\right)$$

pode ser plotada como função de $\Delta R^2$ para diferentes $n$.
Ela revela:

**Para $n$ fixo:** existe um limiar $\Delta R^2_{\min}(n)$ abaixo do qual
o teste não tem poder suficiente para rejeitar $H_0$. Abaixo desse limiar,
o efeito existe mas não é detectável.

**Para $\Delta R^2$ fixo:** à medida que $n$ cresce, o limiar $\Delta R^2_{\min}$
cai e efeitos cada vez menores tornam-se detectáveis. No limite $n \to \infty$,
qualquer $\Delta R^2 > 0$ rejeita $H_0$ — o teste sempre rejeita, tornando
o p-valor uma função apenas de $n$ e não do tamanho do efeito.

$$\Delta R^2_{\min} \approx \frac{F_{\text{crit}} \cdot df_1 \cdot (1 - R^2)}{n}
\xrightarrow{n \to \infty} 0$$

**Implicação:** com $n = 50.000$, o limiar é $\approx 0{,}004\%$. Qualquer
efeito acima disso rejeita $H_0$ formalmente. Por isso o p-valor sozinho é
insuficiente — $\Delta R^2$ é necessário para avaliar relevância prática.

---

## 12. Síntese: O Que as Distribuições Nos Mostram de Verdade

As distribuições de H₀ e H₁ mostram, simultaneamente, três coisas:

**1. Se o resultado é compatível com o acaso** (posição de T_obs na curva azul):
A curva azul define o "mundo onde H₀ é verdadeira". T_obs dentro dessa
curva → os dados são compatíveis com a ausência de efeito. T_obs na cauda
→ os dados contradizem H₀ ao nível α.

**2. O quão extremo é o resultado** (distância de T_obs ao centro de H₀):
Um T_obs que sai do gráfico indica um resultado que seria
praticamente impossível sob H₀ — mesmo antes de calcular o p-valor,
a separação visual já comunica a conclusão.

**3. Se o efeito é real ou artefato de n** (separação entre curvas azul e verde):
Quando $\Delta R^2$ é grande, $\lambda$ é grande, a curva verde está longe
da curva azul e o poder é alto — o efeito é genuíno.
Quando $\Delta R^2$ é pequeno mas $n$ é grande, $\lambda$ é moderado, as
curvas se sobrepõem, o poder é menor — a rejeição de H₀ é artefato amostral.

**A distribuição de H₀ não é sobre os dados — é sobre a estatística.
Ela não descreve pêndulos — descreve o comportamento de F em repetições
hipotéticas do experimento. O que ela nos diz é: se H₀ fosse a lei que
gerou os dados, quanto do que observamos seria esperado por puro acaso.**

---

## Referências

- Fisher, R. A. (1925). *Statistical Methods for Research Workers.* Oliver & Boyd.
- Snedecor, G. W. & Cochran, W. G. (1989). *Statistical Methods* (8ª ed.). Iowa State University Press.
- Cochran, W. G. (1934). *The distribution of quadratic forms in a normal system.* Mathematical Proceedings of the Cambridge Philosophical Society, 30(2).
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data.* MIT Press.
- Lehmann, E. L. & Romano, J. P. (2005). *Testing Statistical Hypotheses* (3ª ed.). Springer.
- Ramsey, J. B. (1969). *Tests for specification errors in classical linear least-squares regression analysis.* JRSS-B, 31(2).

# Explicação da Validação de H2

## $\theta_2$ interfere mais em $\alpha_1$ do que $\omega_1$?

---

## O que H2 pergunta na prática

Tendo em vista um pêndulo duplo. O primeiro elo tem sua própria aceleração angular $\alpha_1$. A pergunta é: para prever **$\alpha_1$**, importa mais saber onde está o segundo elo ($\theta_2$) ou quão rápido está girando o próprio primeiro elo ($\omega_1$)?

Esta hipótese é o espelho simétrico de H3. H3 perguntou se o elo de cima afeta o de baixo; H_4 pergunta se o elo de baixo afeta o de cima

## As duas hipóteses — o que cada uma diz

**$H_0: \beta_{\theta_2} = 0$**

A posição do segundo elo $\theta_2$ não tem influência sobre a aceleração do primeiro elo $\alpha_1$.

**$H_1: \beta_{\theta_2} \neq 0 $**

A posição do segundo elo $\theta_2$ tem alguma influência sobre a aceleração do primeiro $\alpha_1$.

A hipótese complementar $H_0'/H_1'$ vai além: não apenas testa se $\theta_2$ contribui, mas se essa contribuição é *maior* do que a de $\omega_1$. Isso é o "interfere mais" da pergunta original.

---

## Por que dois modelos aninhados?

Não se pode simplesmente olhar para a correlação entre $\theta_2$ e $\alpha_1$ — essa correlação pode existir por causa de variáveis confundidoras que influenciam ambos simultaneamente.

A solução é comparar dois modelos onde a única diferença é a presença ou ausência de $\theta_2$:

$$
M_{\text{sem}}: \quad \hat\alpha_1 = \beta_0 + \beta_1\sin\theta_1 + \beta_2\cos\theta_1 + \beta_3\omega_1
$$

$$
M_{\text{com}}: \quad \hat\alpha_1 = \beta_0 + \beta_1\sin\theta_1 + \beta_2\cos\theta_1 + \beta_3\sin\theta_2 + \beta_4\cos\theta_2 + \beta_5\omega_1
$$

Se $\theta_2$ for útil, $M_{\text{com}}$ cometerá menos erros que $M_{\text{sem}}$. A questão é: essa redução de erros é grande o suficiente para não ter acontecido por acaso?

Obs.: Apesar da aplicação do exato mesmo processo realizado para a hipótese H1, na equação $ M_{\text{com}} $, fora evidenciado neste documento que, para as variáveis $ \theta_1 $ e $ \theta_2 $, houve a segmentação em seus valores trigonométricos de seno e cosseno.

## A estatística F — o que ela calcula

O $RSS$ (*Residual Sum of Squares*) é a soma dos quadrados dos erros de cada modelo — quanto menor, melhor o ajuste.

$$
T = F = \frac{(RSS_{\text{sem}} - RSS_{\text{com}})/q}{RSS_{\text{com}}/(n - k)}
$$

onde $q = 2$ é o número de restrições testadas (os dois coeficientes de $\theta_2$) e $k = 6$ é o número total de parâmetros do modelo completo (intercepto + 5 regressores).

O **numerador** mede a melhora de ajuste ao adicionar $\theta_2$ — o quanto os erros caíram, dividido pelo número de restrições. O **denominador** mede a variabilidade residual média do modelo completo — a escala natural dos erros.

Se $\theta_2$ não contribuísse com nada ($H_0$ verdadeira), adicionar duas variáveis aleatórias ainda melhoraria levemente o ajuste em amostra finita. A estatística $F$ mede se a melhora observada é maior do que o esperado por esse acaso.

---

## A distribuição $F(2,\, 49994)$ — por que essa e não outra?

Sob $H_0$, se os erros forem independentes e com variância constante, essa razão segue exatamente a distribuição $F$ com $q$ graus de liberdade no numerador e $n - k$ no denominador.

O **2** no numerador vem do fato de que você está testando duas restrições simultâneas: $\beta_{\sin\theta_2} = 0$ e $\beta_{\cos\theta_2} = 0$. Por isso o denominador do numerador divide por $q = 2$, normalizando a contribuição média por restrição.

O **49994** no denominador é $n - k = 50000 - 6$, onde $k = 6$ é o número de parâmetros do modelo completo.

A distribuição $F(2, 49994)$ com $n$ tão grande fica muito concentrada próxima de 1 quando $H_0$ é verdadeira. Um $F$ de 12.573 é absurdamente distante desse centro.

---

## O resultado — o que $F = 12573$ significa

Sob $H_0$, você esperaria $F$ próximo de 1. Você observou $F = 12573{,}06$.

$$
p = P(F_{2,\,49994} \geq 12573{,}06 \mid H_0) \approx 0
$$

Isso significa: se $\theta_2$ realmente não contribuísse para $\alpha_1$, a probabilidade de observar uma melhora de ajuste desta magnitude seria praticamente zero. Como esse $p$-valor é menor que $\alpha = 0{,}05$, rejeita-se $H_0$.

O $\Delta R^2 = +0{,}2650$ traduz isso em termos práticos: adicionar $\theta_2$ a um modelo que já tem $\{\theta_1, \omega_1\}$ explica 26,5 pontos percentuais adicionais de variância de $\alpha_1$. O modelo sem $\theta_2$ explicava 20,8% — o modelo com $\theta_2$ explica 47,3%. A melhora relativa é de **127,3%** — mais do que dobrar o poder explicativo do modelo.

---

## A Informação Mútua — por que ela aparece junto?

O F-parcial testa se a contribuição *linear* de $\theta_2$ é significativa. Mas e se a relação for não-linear — como seria de se esperar num sistema caótico como o pêndulo duplo?

A Informação Mútua (MI) captura **qualquer tipo de dependência** — linear, trigonométrica, qualquer outra. Ela mede em nats quanto conhecer $\theta_2$ reduz a incerteza sobre $\alpha_1$, sem assumir forma funcional:

$$
I(X; Y) = \int\int p(x, y)\, \log\frac{p(x, y)}{p(x)\,p(y)}\, dx\, dy
$$

Para a MI, os ângulos foram reconstruídos como escalares via $\theta = \text{arctan2}(\sin\theta, \cos\theta)$, preservando a geometria circular.

| Variável    | MI com$\alpha_1$   |
| ------------ | -------------------- |
| $\theta_2$ | **0,832 nats** |
| $\omega_1$ | 0,307 nats           |

$\theta_2$ tem $2{,}71\times$ mais informação sobre $\alpha_1$ do que $\omega_1$. Isso é consistente com o F-parcial e responde diretamente à hipótese complementar $H_0'/H_1'$: a contribuição de $\theta_2$ é maior do que a de $\omega_1$ não apenas no sentido linear, mas em qualquer tipo de dependência estatística.

---

## Comparação com H3 — o que os resultados juntos revelam

| Métrica              | H1:$\theta_1 \to \alpha_2$ | H2:$\theta_2 \to \alpha_1$ |
| --------------------- | ---------------------------- | ---------------------------- |
| $F$                 | 1895,80                      | **12573,06**           |
| $\Delta R^2$        | +0,029                       | **+0,265**             |
| MI do ângulo cruzado | 0,785 nats                   | **0,832 nats**         |
| Razão MI             | 1,72×                       | **2,71×**             |

O acoplamento reverso ($\theta_2 \to \alpha_1$) é **substancialmente mais forte** do que o direto ($\theta_1 \to \alpha_2$). Isso faz sentido físico: o segundo elo está em balanço livre e suas forças de inércia são transmitidas integralmente para o primeiro elo através do ponto de junção. O primeiro elo, por sua vez, está preso no pivot fixo, o que atenua a propagação inversa.

---

## Por que "erro sistemático irredutível" na conclusão?

Se você treinar um modelo que só recebe $\{\theta_1, \omega_1\}$ para prever $\alpha_1$, ele estará omitindo $\theta_2$ deliberadamente. O teste mostrou que $\theta_2$ carrega $0{,}832$ nats de informação sobre $\alpha_1$ que as outras variáveis não fornecem.

Essa informação simplesmente não estará disponível para o modelo. Não importa o quão sofisticada seja a rede, quantos dados você use ou como ajuste os hiperparâmetros — o erro não pode cair abaixo de um patamar determinado pela ausência de $\theta_2$ na entrada. O efeito aqui é ainda mais severo do que em H3: com $\Delta R^2 = 0{,}265$, a omissão de $\theta_2$ deixa 26,5% da variância de $\alpha_1$ completamente inexplicável por construção.

Por isso a implicação é arquitetural: a entrada do modelo precisa ter o estado de **ambos** os elos, e a rede precisa ser projetada para processar o estado conjunto do sistema.

---

## Resumo

| Elemento                          | Valor                                               | Interpretação                                              |
| --------------------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| $H_0$                           | $\beta_{\sin\theta_2} = \beta_{\cos\theta_2} = 0$ | $\theta_2$ não contribui para $\alpha_1$                |
| Estatística$T$                 | $F = 12573{,}06$                                  | Melhora de ajuste ao adicionar$\theta_2$                   |
| Distribuição sob$H_0$         | $F(2,\, 49994)$                                   | Duas restrições,$n - 6$ graus de liberdade               |
| $p$-valor                       | $\approx 0$                                       | $P(F \geq 12573{,}06 \mid H_0)$                            |
| Decisão ($\alpha = 0{,}05$)    | **Rejeita $H_0$**                           | $p \approx 0 < 0{,}05$                                     |
| $\Delta R^2$                    | $+0{,}2650$                                       | Ganho incremental de$\theta_2$ sobre o modelo restrito     |
| $\text{MI}(\theta_2, \alpha_1)$ | $0{,}832$ nats                                    | $2{,}71\times$ maior que $\text{MI}(\omega_1, \alpha_1)$ |
| Implicação                      | Entrada conjunta obrigatória                       | Modelos de elo único são sistematicamente inferiores       |

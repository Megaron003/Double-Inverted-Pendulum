# Explicação da Validação de H1

## $\theta_1$ interfere mais em $\alpha_2$ do que $\omega_2$?

---

## O que H1 pergunta na prática

Você tem um pêndulo duplo. O segundo elo tem sua própria aceleração angular $\alpha_2$. A pergunta é: para prever **$\alpha_2$, **importa mais saber onde está o primeiro elo ($\theta_1$) ou quão rápido está girando o segundo elo ($\omega_2$)?

Fisicamente isso não é óbvio. $\theta_1$ é o ângulo de um elo diferente — por que ele afetaria a aceleração do outro? A hipótese diz que sim, e o teste mede o quanto.

---

## As duas hipóteses — o que cada uma diz

**$H_0: \beta_{\theta_1} = 0$**

Traduzindo literalmente: "no modelo que já usa $\theta_2$, $\omega_1$ e $\omega_2$ para prever $\alpha_2$, adicionar $\theta_1$ não muda nada". O coeficiente de $\theta_1$ seria zero — $\theta_1$ não acrescenta nenhuma informação além do que as outras três variáveis já fornecem.

**$H_1: \beta_{\theta_1} \neq 0$**

"$\theta_1$ acrescenta algo que $\{\theta_2, \omega_1, \omega_2\}$ não conseguem explicar sozinhos". Isso seria evidência de acoplamento dinâmico real entre os elos.

A hipótese complementar $H_0'$ e $H_1'$ vai além: não apenas testa se $\theta_1$ contribui, mas se essa contribuição é *maior* do que a de $\omega_2$. Isso é o "interfere mais" da pergunta original.

---

## Por que dois modelos aninhados?

Não pode-se simplesmente olhar para a correlação entre $\theta_1$ e $\alpha_2$ e concluir que $\theta_1$ é importante — essa correlação pode existir por causa de uma terceira variável que influencia ambos.

A solução é comparar dois modelos onde a única diferença é a presença ou ausência de $\theta_1$:

$$
M_{\text{sem}}: \quad \hat\alpha_2 = \beta_0 + \beta_1\theta_2 + \beta_2\omega_1 + \beta_3\omega_2
$$

$$
M_{\text{com}}: \quad \hat\alpha_2 = \beta_0 + \beta_1\theta_1 + \beta_2\theta_2 + \beta_3\omega_1 + \beta_4\omega_2
$$

Se $\theta_1$ for útil, $M_{\text{com}}$ cometerá menos erros que $M_{\text{sem}}$. A questão é: essa redução de erros é grande o suficiente para não ter acontecido por acaso?

---

## A estatística F — o que ela calcula

O $RSS$ (*Residual Sum of Squares*) é a soma dos quadrados dos erros de cada modelo — quanto menor, melhor o ajuste.

$$
T = F = \frac{(RSS_{\text{sem}} - RSS_{\text{com}})/1}{RSS_{\text{com}}/(n - 5)}
$$

O **numerador** mede a melhora de ajuste ao adicionar $\theta_1$, isto é, o quanto os erros caíram. O **denominador** mede a variabilidade residual média do modelo completo, o que define a escala natural dos erros.

A divisão responde: "a melhora de ajuste é grande em relação ao ruído residual do modelo?"

Se $\theta_1$ não contribuísse com nada ($H_0$ verdadeira), adicionar uma variável aleatória qualquer ainda melhoraria levemente o ajuste em amostra finita — isso é normal e esperado. A estatística $F$ mede se a melhora observada é maior do que o esperado por esse acaso.

---

## A distribuição $F(1, 49995)$ — por que essa e não outra?

Sob $H_0$, se os erros forem independentes e com variância constante, essa razão segue exatamente a distribuição $F$ com 1 grau de liberdade no numerador e $n - 5$ no denominador.

O **1** no numerador vem do fato de que você está testando uma única restrição: $\beta_{\theta_1} = 0$. Se testasse duas variáveis ao mesmo tempo, seria $F(2, n-k)$.

O **49995** no denominador é $n - k - 1 = 50000 - 4 - 1$, onde $k = 4$ é o número de regressores do modelo completo.

A distribuição $F(1, 49995)$ com $n$ tão grande fica muito concentrada próxima de 1 quando $H_0$ é verdadeira. Um $F$ de 1.895 é absurdamente distante desse centro.

---

## O resultado — o que $F = 1895$ significa

Sob $H_0$, você esperaria $F$ próximo de 1. Você observou $F = 1895{,}80$.

$$
p = P(F_{1,\,49995} \geq 1895{,}80 \mid H_0) \approx 0
$$

Isso significa: se $\theta_1$ realmente não contribuísse para $\alpha_2$, a probabilidade de observar uma melhora de ajuste desta magnitude seria praticamente zero. Como esse p-valor é menor que $\alpha = 0{,}05$, rejeita-se $H_0$.

O $\Delta R^2 = +0{,}0289$ traduz isso em termos práticos: adicionar $\theta_1$ a um modelo que já tem $\{\theta_2, \omega_1, \omega_2\}$ explica 2,9 pontos percentuais adicionais de variância de $\alpha_2$. O modelo sem $\theta_1$ explicava 20,9% — o modelo com $\theta_1$ explica 23,8%. A melhora relativa é de 13,8%.

---

## A Informação Mútua — por que ela aparece junto?

O F-parcial testa se a contribuição *linear* de $\theta_1$ é significativa. Mas e se a relação for não-linear?

A Informação Mútua (MI) captura **qualquer tipo de dependência** — linear, trigonométrica, qualquer outra. Ela mede em nats quanto conhecer $\theta_1$ reduz a incerteza sobre $\alpha_2$, sem assumir forma funcional:

$$
I(X; Y) = \int\int p(x, y)\, \log\frac{p(x, y)}{p(x)\,p(y)}\, dx\, dy
$$

| Variável    | MI com$\alpha_2$   |
| ------------ | -------------------- |
| $\theta_1$ | **0,785 nats** |
| $\omega_2$ | 0,457 nats           |

$\theta_1$ tem $1{,}72\times$ mais informação sobre $\alpha_2$ do que $\omega_2$. Isso é consistente com o F-parcial e responde diretamente à hipótese complementar $H_0'/H_1'$: a contribuição de $\theta_1$ é maior do que a de $\omega_2$ não apenas no sentido linear, mas em qualquer tipo de dependência estatística.

---

## Por que "erro sistemático irredutível" na conclusão?

Se você treinar um modelo que só recebe $\{\theta_2, \omega_2\}$ para prever $\alpha_2$, ele estará omitindo $\theta_1$ deliberadamente. O teste mostrou que $\theta_1$ carrega $0{,}785$ nats de informação sobre $\alpha_2$ que as outras variáveis não fornecem.

Essa informação simplesmente não estará disponível para o modelo. Não importa o quão sofisticada seja a rede, quantos dados você use ou como ajuste os hiperparâmetros — o erro não pode cair abaixo de um patamar determinado pela ausência de $\theta_1$ na entrada. É um teto imposto pela especificação do modelo, não pela capacidade de aprendizado.

Por isso a implicação é arquitetural: a entrada do modelo precisa ter todos os elos, e a rede precisa ser projetada para processar o estado conjunto do sistema.

---

## Resumo

| Elemento                          | Valor                         | Interpretação                                              |
| --------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| $H_0$                           | $\beta_{\theta_1} = 0$      | $\theta_1$ não contribui para $\alpha_2$                |
| Estatística$T$                 | $F = 1895{,}80$             | Melhora de ajuste ao adicionar$\theta_1$                   |
| Distribuição sob$H_0$         | $F(1,\, 49995)$             | Uma restrição,$n - 5$ graus de liberdade                 |
| $p$-valor                       | $\approx 0$                 | $P(F \geq 1895{,}80 \mid H_0)$                             |
| Decisão ($\alpha = 0{,}05$)    | **Aceita $H_0$**      | $p \approx 0 < 0{,}05$                                     |
| $\Delta R^2$                    | $+0{,}0289$                 | Ganho incremental de$\theta_1$ sobre o modelo restrito     |
| $\text{MI}(\theta_1, \alpha_2)$ | $0{,}785$ nats              | $1{,}72\times$ maior que $\text{MI}(\omega_2, \alpha_2)$ |
| Implicação                      | Entrada conjunta obrigatória | Modelos de elo único são sistematicamente inferiores       |

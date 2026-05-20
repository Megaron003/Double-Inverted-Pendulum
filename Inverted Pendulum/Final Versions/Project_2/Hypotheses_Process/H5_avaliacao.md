# Explicação da Validação de HN-F
## sin θ₂ ou cos θ₂: qual componente explica α₁?

---

## O que H5 pergunta na prática

H2 confirmou que θ₂ é indispensável para prever α₁ — com ΔR² = 26,5%,
omitir θ₂ deixa mais de um quarto da variância inexplicável. Mas θ₂ entra
no modelo como dois termos: $\sin\theta_2$ e $\cos\theta_2$. A pergunta
de HN-F é: essa contribuição vem igualmente dos dois, ou um domina?

A resposta tem implicação direta na representação das features: se $\sin\theta_2$
domina, ele é a feature prioritária para qualquer modelo que tente aprender
$f(\mathbf{x}) \to \dot{\mathbf{x}}$.

---

## As duas hipóteses — o que cada uma diz

**$H_0: \beta_{\sin\theta_2} = 0$ dado $\cos\theta_2$**

$\sin\theta_2$ é redundante quando $\cos\theta_2$ já está presente no modelo.
Todo o poder preditivo de θ₂ viria de $\cos\theta_2$.

**$H_1: \beta_{\sin\theta_2} \neq 0$ dado $\cos\theta_2$**

$\sin\theta_2$ acrescenta informação além do que $\cos\theta_2$ já captura.

Simetricamente:

**$H_0': \beta_{\cos\theta_2} = 0$ dado $\sin\theta_2$**

$\cos\theta_2$ é redundante quando $\sin\theta_2$ já está presente.

**$H_1': \beta_{\cos\theta_2} \neq 0$ dado $\sin\theta_2$**

$\cos\theta_2$ acrescenta além de $\sin\theta_2$.

---

## Por que quatro modelos aninhados?

Para isolar a contribuição de cada componente controlando pela presença
do outro, constroem-se quatro modelos:

$$M_{\text{full}}: \quad \hat\alpha_1 = \beta_0 + \beta_1\sin\theta_1 + \beta_2\cos\theta_1 + \beta_3\sin\theta_2 + \beta_4\cos\theta_2 + \beta_5\omega_1$$

$$M_{\text{sem\_sin}}: \quad \hat\alpha_1 = \beta_0 + \beta_1\sin\theta_1 + \beta_2\cos\theta_1 + \beta_3\cos\theta_2 + \beta_4\omega_1$$

$$M_{\text{sem\_cos}}: \quad \hat\alpha_1 = \beta_0 + \beta_1\sin\theta_1 + \beta_2\cos\theta_1 + \beta_3\sin\theta_2 + \beta_4\omega_1$$

$$M_{\text{sem\_tudo}}: \quad \hat\alpha_1 = \beta_0 + \beta_1\sin\theta_1 + \beta_2\cos\theta_1 + \beta_3\omega_1$$

$M_{\text{sem\_tudo}}$ é exatamente o modelo restrito de H2 — o ponto de partida.

---

## A estatística F — o que ela calcula

Para testar a contribuição de $\sin\theta_2$ dado que $\cos\theta_2$ já está presente:

$$T = F = \frac{(RSS_{M_{\text{sem\_sin}}} - RSS_{M_{\text{full}}})/1}{RSS_{M_{\text{full}}}/(n - k)}$$

onde $q = 1$ é o número de restrições testadas (apenas $\beta_{\sin\theta_2} = 0$)
e $k = 6$ é o número de parâmetros do modelo completo.

---

## A distribuição $F(1,\, 49994)$

Sob $H_0$, essa razão segue $F(1, 49994)$. O **1** no numerador reflete
que se testa uma única restrição. Com $n = 50.000$, a distribuição
fica extremamente concentrada próxima de 1 — qualquer efeito genuíno
produz $F$ muito distante desse centro.

---

## O resultado — o que $F = 25137$ e $F = 3{,}96$ significam

Para $\sin\theta_2$:

$$p = P(F_{1,\,49994} \geq 25137{,}52 \mid H_0) \approx 0$$

Para $\cos\theta_2$:

$$p = P(F_{1,\,49994} \geq 3{,}96 \mid H_0) = 0{,}0465$$

O primeiro rejeita $H_0$ com $p \approx 0$ e $F = 25.137$.
O segundo fica no limiar — $p = 0{,}0465 < \alpha = 0{,}05$ formalmente
rejeita, mas $F = 3{,}96$ está apenas marginalmente acima de $F_{\text{crit}} = 3{,}842$.

O $\Delta R^2$ traduz isso em termos práticos:

| Componente | ΔR² | % do total |
|------------|-----|------------|
| sin θ₂ dado cos θ₂ | 0,264912 | 99,97% |
| cos θ₂ dado sin θ₂ | 0,000042 | 0,02% |
| Conjunto (ambos) | 0,265002 | 100% |

$\sin\theta_2$ sozinho explica praticamente toda a contribuição de θ₂.
$\cos\theta_2$ adiciona 0,004 pontos percentuais de $R^2$ — efeito que
só aparece como significativo por causa do $n$ grande (paradoxo do
tamanho amostral, já discutido em HN-A).

---

## A Informação Mútua — confirmação não-linear

| Variável | MI com α₁ |
|----------|-----------|
| sin θ₂ | 0,7855 nats |
| cos θ₂ | 0,0493 nats |
| Razão | 15,9× |

A MI confirma o F-parcial por um critério independente e sem assumir
linearidade: $\sin\theta_2$ carrega 16× mais informação sobre $\alpha_1$
do que $\cos\theta_2$.

---

## Por que sin θ₂ e não cos θ₂?

A resposta está nas equações de movimento do pêndulo duplo. O termo de
força gravitacional sobre o segundo elo nas equações de Lagrange contém:

$$(m_1 + m_2)\,g\,\sin\theta_2$$

Este é o acoplamento dominante que aparece na aceleração do primeiro
elo via reação no ponto de junção. O termo $\cos\theta_2$ aparece nos
termos de inércia e força centrífuga — efeitos de segunda ordem neste
regime de operação.

O resultado empírico — $\sin\theta_2$ explicando 99,97% da contribuição
de θ₂ — é exatamente o que as equações de Lagrange predizem.

---

## Distinção em relação a H2

H2 testou: *"θ₂ como um todo contribui para α₁?"*

HN-F testa: *"dentro de θ₂, qual componente trigonométrico carrega
essa contribuição?"*

A relação entre as duas é hierárquica: HN-F é uma decomposição interna
da hipótese confirmada por H2. Ela não contradiz H2 — especifica de
onde vem o efeito.

---

## Implicação arquitetural

A feature mais informativa para prever $\alpha_1$ a partir de θ₂ é
$\sin\theta_2$, não θ₂ bruto nem $\cos\theta_2$.

Isso tem consequência direta na camada de entrada da rede:

Fornecer $\sin\theta_2$ explicitamente significa que a rede não precisa
aprender internamente a transformação $\theta_2 \to \sin\theta_2$, o que
reduziria a profundidade necessária e aceleraria a convergência.

Uma rede que receba $\theta_2$ bruto consegue aprender sin por meio de
composição de ativações tanh — mas isso custa camadas e neurônios extras
para uma transformação que já é conhecida a priori.

---

## Resumo

| Elemento | sin θ₂ | cos θ₂ |
|----------|--------|--------|
| H₀ testada | β_sinθ₂ = 0 dado cosθ₂ | β_cosθ₂ = 0 dado sinθ₂ |
| F-parcial | 25137,52 | 3,96 |
| Distribuição sob H₀ | F(1, 49994) | F(1, 49994) |
| p-valor | ≈ 0 | 0,0465 |
| Decisão (α=0,05) | Rejeita H₀ ✓ | Rejeita H₀ (limiar) |
| ΔR² | 0,264912 | 0,000042 |
| % da contribuição total | 99,97% | 0,02% |
| IC 99% do coeficiente | [+25,33 ; +26,17] não contém 0 ✓ | [−0,11 ; +0,83] contém 0 ✗ |
| MI com α₁ | 0,7855 nats | 0,0493 nats |
| Feature prioritária | Sim ✓ | Não ✗ |

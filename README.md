# polymarket-mm-sim

Backtest de stratégies de market making sur les marchés de prédiction Polymarket — **sans clé API, sans wallet et sans capital**.

Le projet part du [keeper officiel de Polymarket](https://github.com/Polymarket/poly-market-maker) et lui ajoute la pièce qui lui manque : un environnement de simulation. Les stratégies amont sont conservées **telles quelles** ; seules les deux dépendances externes — l'API CLOB et la blockchain — sont remplacées par un marché synthétique et un moteur d'exécution local.

## Pourquoi

Le keeper officiel ne sait tourner qu'en production : il lui faut une clé privée, du USDC sur Polygon et un marché réel. Impossible, donc, de répondre hors ligne à des questions pourtant élémentaires — quelle stratégie résiste le mieux à la volatilité ? à partir de quel niveau de frais devient-elle non rentable ? quel déséquilibre d'inventaire faut-il tolérer ?

Ce dépôt rend ces questions mesurables, et le premier résultat est arrivé sans qu'on le cherche : **le keeper amont plante sur un marché qui dérive vers sa résolution** (voir plus bas).

## Installation

```bash
git clone https://github.com/louisjeromejamin-pixel/polymarket-mm-sim.git
cd polymarket-mm-sim
pip install -r requirements-sim.txt   # simulation seule : aucune dépendance lourde
```

La simulation ne requiert **que la bibliothèque standard**. Le fichier `requirements.txt` d'origine (`web3`, `py-clob-client`…) n'est nécessaire que pour faire tourner le keeper en conditions réelles.

## Utilisation

### Un backtest

```bash
python -m simulation.cli backtest --strategy amm --steps 300 --seed 42
```

```
Backtest — stratégie AMM

  strategie         amm
  pas               300
  valeur initiale   1500.0
  valeur finale     1426.25
  pnl               -73.75
  rendement pct     -4.916
  executions        344
  volume            1245.66
  drawdown max pct  5.977
  sharpe            -0.0759
  inventaire max    314.2
```

### Comparer les deux stratégies

```bash
python -m simulation.cli compare --seeds 25 --steps 300
```

### Balayer un paramètre

```bash
python -m simulation.cli sensitivity --strategy bands --param volatility --values 0.05,0.10,0.15,0.25
python -m simulation.cli sensitivity --strategy bands --param maker_fee --values 0,0.001,0.005,0.01
```

### Depuis Python

```python
from simulation import run_sweep, MarketConfig

results = run_sweep(
    "bands", "config/bands.json",
    seeds=range(50), steps=500,
    market_config=MarketConfig(volatility=0.2, drift=0.01),
)
print(sum(r.pnl for r in results) / len(results))
```

## Comment le marché est simulé

Un contrat binaire se résout à 0 ou 1 ; son prix, borné dans `(0, 1)`, **est** une probabilité. Une marche aléatoire ordinaire conviendrait mal : elle sortirait de l'intervalle, et donnerait la même volatilité absolue à un contrat coté 0,50 qu'à un contrat coté 0,02 — alors qu'un marché quasi résolu ne bouge presque plus.

Le prix évolue donc en **log-odds** :

```
logit(p) = log(p / (1 − p))
```

La marche aléatoire est libre sur tout l'axe réel, et la sigmoïde ramène le prix dans `(0, 1)` par construction, en compressant naturellement les mouvements près des bornes.

**Le modèle d'exécution** reste volontairement pessimiste, pour ne pas fabriquer de performance fictive :

- un ordre n'est touché que si le prix vient effectivement le chercher ;
- même touché, il n'est servi qu'avec une probabilité `fill_probability` (0,6 par défaut) — sur un vrai carnet, être au bon prix ne garantit pas d'être en tête de file ;
- les exécutions sont **partielles** ;
- un achat sans collatéral suffisant ou une vente à découvert sont **refusés**.

L'ordre des opérations dans la boucle est ce qui compte le plus : la stratégie place ses ordres, **puis** le marché bouge, **puis** on confronte. L'inverse ne remplirait jamais rien, puisque la stratégie recentre ses ordres autour du prix courant à chaque pas.

## Résultats

### AMM contre Bands

25 graines, 300 pas, volatilité 0,15, capital initial 1 500 USDC :

| Métrique | AMM | Bands |
|---|---:|---:|
| PnL moyen | −36,39 | **−18,70** |
| PnL médian | −36,00 | −19,77 |
| Écart-type du PnL | 73,11 | **27,54** |
| Pire run | −227,58 | **−76,34** |
| Meilleur run | **70,82** | 31,48 |
| Runs gagnants | 24 % | 28 % |
| Exécutions | 490,4 | 268,6 |
| Drawdown moyen | 6,57 % | **2,80 %** |
| Inventaire max | 485,4 | **172,1** |

**Bands domine sur le risque, pas sur le rendement.** Elle perd deux fois moins, avec un écart-type presque trois fois inférieur, et surtout un inventaire maîtrisé : 172 parts de déséquilibre contre 485 pour l'AMM, qui accumule jusqu'à engager la quasi-totalité de sa position d'un seul côté. En contrepartie, elle négocie deux fois moins et plafonne plus bas sur ses meilleurs runs.

**Les deux stratégies perdent de l'argent en moyenne.** Ce n'est pas un défaut du simulateur : c'est la **sélection adverse**, le risque structurel du métier. Un market maker achète quand le prix descend et vend quand il monte ; sur une martingale sans dérive, il se retrouve systématiquement du mauvais côté du mouvement. En production, ce coût est compensé par le spread capturé sur le flux non informé et par les incitations à la liquidité — deux choses que ce simulateur ne modélise pas. Les chiffres servent donc à **comparer des stratégies entre elles**, pas à prédire une rentabilité.

### Sensibilité à la volatilité (Bands, 20 graines)

| Volatilité | PnL moyen | Écart-type | Runs gagnants |
|---:|---:|---:|---:|
| 0,05 | −6,43 | 11,92 | 25 % |
| 0,10 | −10,18 | 24,27 | 40 % |
| 0,15 | −20,25 | 28,59 | 25 % |
| 0,25 | −34,74 | 28,13 | 15 % |

La perte croît de façon monotone avec la volatilité : c'est la signature attendue de la sélection adverse — plus le marché bouge, plus le market maker est pris à contre-pied.

### Sensibilité aux frais (Bands, 20 graines)

| Frais maker | PnL moyen | Runs gagnants |
|---:|---:|---:|
| 0 % | −20,25 | 25 % |
| 0,1 % | −20,95 | 25 % |
| 0,5 % | −23,71 | 20 % |
| 1 % | −27,17 | 15 % |

Polymarket ne prélève pas de frais maker aujourd'hui. S'ils étaient introduits à 1 %, ils coûteraient ici ~7 USDC sur 300 pas — un tiers de la perte moyenne. Une stratégie qui ne serait que marginalement rentable n'y survivrait pas.

## Le bug trouvé dans le keeper officiel

En poussant la volatilité, le backtest a fait remonter une exception dans la stratégie AMM amont :

```
File "poly_market_maker/strategies/amm.py", line 107, in phi
    return (1 / (sqrt(self.p_i) - sqrt(self.p_l))) * (...)
ZeroDivisionError: float division by zero
```

Quand le prix atteint une borne de la plage configurée (`p_min` = 0,05 ou `p_max` = 0,95), `set_price()` produit `p_l == p_i` et une liste `buy_prices` vide. Les deux cas dégénèrent : division par zéro d'un côté, `IndexError` sur `buy_prices[0]` de l'autre.

Ce n'est pas un cas de laboratoire — **tout marché de prédiction dérive vers 0 ou 1 en approchant de sa résolution**. Un keeper laissé tourner sur un marché qui se dénoue rencontre nécessairement cette situation.

Le correctif ([`amm.py`](poly_market_maker/strategies/amm.py)) renvoie une allocation nulle dans ces deux cas : quand aucun achat n'est possible de ce côté, il n'y a pas de collatéral à lui allouer. Un test ([`test_backtest.py`](tests/test_backtest.py)) vérifie qu'aucune erreur de stratégie ne remonte sur 400 pas à volatilité 0,5.

## Structure

```
poly_market_maker/       # code amont (Polymarket, MIT) — stratégies inchangées
├── strategies/
│   ├── amm.py           # + correctif des cas dégénérés dans phi()
│   ├── bands.py
│   └── ...
├── order.py             # + import py_clob_client rendu optionnel
└── utils.py             # + imports web3/yaml rendus optionnels
simulation/              # ajouts de ce dépôt
├── market_sim.py        # marché binaire en log-odds
├── execution.py         # carnet, exécutions partielles, portefeuille
├── backtest.py          # boucle de backtest et métriques
└── cli.py               # backtest / compare / sensitivity
tests/                   # 38 tests
config/                  # configurations amont des stratégies
docs/                    # documentation amont des stratégies
```

Les trois modifications du code amont sont minimales et signalées en commentaire. Les deux dernières servent uniquement à faire tourner les stratégies hors ligne : sans elles, importer un `Order` exigeait d'installer tout `web3`.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

Les tests couvrent l'aller-retour logit/sigmoïde, le maintien du prix dans `(0, 1)` sous volatilité extrême, le sens des exécutions (un achat sous le marché ne passe pas ; le marché doit venir le chercher), la complémentarité des prix des deux tokens, le refus des ventes à découvert et des achats sans collatéral, la reproductibilité à graine fixée, et la non-régression du correctif AMM.

## Limites

- **Le marché ignore le market maker.** Ses ordres ne déplacent pas le prix, ce qui surestime la performance à taille importante.
- **Pas de flux informé/non informé.** Un vrai carnet mêle des contreparties bruitées, dont le market maker tire son revenu. Ici, toute contrepartie est effectivement informée — d'où des PnL négatifs par construction.
- **Aucune incitation à la liquidité**, alors qu'elles constituent une part majeure du revenu réel sur Polymarket.
- **Le pas de simulation n'a pas de durée calendaire**, donc le Sharpe affiché n'est pas annualisé et ne se compare qu'entre runs de ce simulateur.

## Prochaines étapes

- Rejouer des carnets **historiques** réels via l'API publique Polymarket, plutôt qu'un marché synthétique.
- Modéliser deux flux de contreparties (informé / non informé) pour rendre le market making rentable et mesurer le point d'équilibre.
- Ajouter une stratégie à **skew d'inventaire** : décaler les cotations en fonction de la position pour ramener activement l'inventaire vers zéro.
- Simuler l'impact de marché des ordres du keeper.

## Crédits et licence

Le répertoire `poly_market_maker/` provient de [Polymarket/poly-market-maker](https://github.com/Polymarket/poly-market-maker), distribué sous licence MIT — copyright (c) 2023 Polymarket. La licence d'origine est conservée dans [LICENSE](LICENSE).

Les ajouts de ce dépôt (`simulation/`, `tests/`) et les correctifs apportés au code amont sont publiés sous la même licence MIT.

> **Avertissement.** Ce projet est un outil d'étude. Il ne constitue pas un conseil en investissement, et les résultats de simulation ne préjugent en rien de performances réelles. Le code de trading en production du dépôt amont est expérimental — l'utiliser avec du capital réel est à vos risques.

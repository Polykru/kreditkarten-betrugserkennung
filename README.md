# Kreditkarten-Betrugserkennung

In diesem Projekt habe ich ein Modell trainiert, das Kreditkartenbetrug erkennen soll. Ich wollte vor allem verstehen, wie man mit Daten umgeht, in denen Betrug nur sehr selten vorkommt, und wie man so ein Modell richtig bewertet.

Verwendet: Python, pandas, scikit-learn, matplotlib, Google Colab

## Worum geht es?

Banken können nicht jede Kartenzahlung von Hand prüfen. Deshalb sollen verdächtige Zahlungen automatisch erkannt werden. Dabei kann ein Modell zwei Fehler machen:

1. Es übersieht einen Betrug – dann verliert die Bank Geld.
2. Es blockiert einen ehrlichen Kunden – dann ist der Kunde verärgert.

Man kann nicht beide Fehler gleichzeitig auf null bringen. In meinem Projekt wollte ich zeigen, wo ein guter Kompromiss liegt.

## Daten

Ich habe den Datensatz [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) von Kaggle benutzt (Université Libre de Bruxelles). Er enthält 284.807 Kartenzahlungen aus Europa, die innerhalb von ungefähr zwei Tagen gemacht wurden. Davon sind nur 492 Betrug, also 0,17 %.

Die Spalten:
- `Time`: Sekunden seit der ersten Zahlung im Datensatz
- `Amount`: Betrag
- `V1` bis `V28`: Merkmale, die aus Datenschutzgründen anonymisiert wurden (mit PCA). Man weiß also nicht, was sie genau bedeuten.
- `Class`: 0 = normal, 1 = Betrug

## Was mir bei den Daten aufgefallen ist

Es gibt keine fehlenden Werte, aber 1.081 doppelte Zeilen. Die habe ich gelöscht, damit dieselbe Zahlung nicht gleichzeitig im Training und im Test landet.

Interessant war der Vergleich der Beträge: Im Durchschnitt sind Betrugszahlungen höher (122 € gegenüber 88 €), der Median ist aber viel niedriger (9,25 € gegenüber 22 €). Ein Viertel der Betrugsfälle liegt bei höchstens 1 €. Das passt zu „Card Testing“: Betrüger testen eine gestohlene Karte zuerst mit einem kleinen Betrag. Man sieht daran auch, dass man Betrug nicht nur am Betrag erkennen kann.

## Warum Accuracy hier nichts bringt

Wenn ein Modell einfach immer „kein Betrug“ sagt, liegt es in 99,83 % der Fälle richtig. Trotzdem findet es keinen einzigen Betrug. Deshalb schaue ich mir zwei andere Werte an:

- Recall: Wie viele Betrugsfälle findet das Modell?
- Precision: Wie viele der Alarme sind wirklich Betrug?

## Wie ich vorgegangen bin

1. Duplikate entfernt
2. Daten in 80 % Training und 20 % Test aufgeteilt, mit `stratify`, damit in beiden Teilen gleich viel Betrug ist
3. `Time` und `Amount` skaliert. Den Scaler habe ich nur auf den Trainingsdaten angepasst, damit keine Informationen aus dem Test ins Training kommen (Data Leakage).
4. Zwei Modelle trainiert: Logistic Regression und Random Forest
5. Verschiedene Schwellenwerte ausprobiert
6. Precision-Recall-Kurve erstellt

## Ergebnisse

Im Testset sind 56.746 Zahlungen, davon 95 Betrugsfälle.

| Modell | Recall | Precision | gefunden | übersehen | Fehlalarme |
|---|---|---|---|---|---|
| Logistic Regression | 59 % | 85 % | 56 | 39 | 10 |
| Random Forest | 73 % | 97 % | 69 | 26 | 2 |

Der Random Forest war bei beiden Werten besser. Er findet mehr Betrug und blockiert trotzdem fast keine ehrlichen Kunden.

### Schwellenwert

Normalerweise sagt das Modell „Betrug“, sobald die Wahrscheinlichkeit über 50 % liegt. Ich habe ausprobiert, was passiert, wenn man diese Grenze senkt:

| Schwellenwert | Recall | Precision | Fehlalarme |
|---|---|---|---|
| 0,5 | 73 % | 97 % | 2 |
| 0,4 | 76 % | 96 % | 3 |
| 0,3 | 77 % | 92 % | 6 |
| 0,2 | 77 % | 86 % | 12 |
| 0,1 | 78 % | 76 % | 24 |

Ich würde 0,4 nehmen. Damit findet das Modell drei Betrugsfälle mehr und es gibt nur einen Fehlalarm mehr. Darunter lohnt es sich kaum noch: Der Recall steigt fast nicht mehr, aber es werden immer mehr ehrliche Kunden blockiert. Am Ende muss das aber die Bank entscheiden, je nachdem, was für sie teurer ist.

### Precision-Recall-Kurve

Bis zu einem Recall von etwa 75 % macht das Modell fast keine Fehler, danach fällt die Kurve steil ab. Die Average Precision liegt bei 0,79. Ein Modell, das nur rät, käme auf ungefähr 0,0017.

## Grenzen

- Ungefähr 20 Betrugsfälle findet das Modell auch bei einem sehr niedrigen Schwellenwert nicht. Für das Modell sehen diese Zahlungen ganz normal aus.
- Weil V1 bis V28 anonymisiert sind, kann ich nicht sagen, welche Eigenschaften genau auf Betrug hinweisen.
- Die Daten sind nur von zwei Tagen im Jahr 2013.

## Was ich als Nächstes ausprobieren würde

- XGBoost als weiteres Modell
- `class_weight="balanced"` oder SMOTE, damit das Modell die seltenen Betrugsfälle stärker beachtet
- Kreuzvalidierung, um die Ergebnisse besser abzusichern

## Was ich gelernt habe

Ich habe zum ersten Mal Schritt für Schritt einen echten Datensatz analysiert und gesehen, in welcher Reihenfolge man so ein Projekt angeht. Dabei habe ich gemerkt, wie viele kleine Dinge man beachten muss, zum Beispiel Duplikate oder Data Leakage. Und dass die Wahl des Schwellenwerts am Ende keine rein technische Entscheidung ist, sondern eine für das Unternehmen.

## Ausführen

1. `creditcard.csv` von Kaggle herunterladen
2. `betrugserkennung.ipynb` in Google Colab öffnen
3. Pfad zur CSV-Datei anpassen und alle Zellen ausführen

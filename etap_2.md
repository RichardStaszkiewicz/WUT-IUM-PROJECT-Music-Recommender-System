# IUM 
## System rekomendacji utworów

### Poniższy raport przedstawia:
- opis wykorzystywanych modeli
- trening modeli
- porównanie wyników modeli
- implementację mikroserwisu
- testy A/B

### Model 1 - Autoenkoder

Opis modelu znajduje się w pliku `model1_ae.ipynb`.

Autoenkoder składa się z kodera i dekodera.
Koder uczy się kodować dane wejściowe do reprezentacji przestrzeni ukrytej
o wyższym wymiarze, podczas gdy dekoder rekonstruuje oryginalne dane z przestrzeni ukrytej.

Po wytrenowaniu autoenkodera wykorzystywana jest część kodera do zakodowania utworu,
którego rekomendacje są poszukiwane, przechwytując jej charakterystykę zbudowaną
przez model w przestrzeni ukrytej.
Porównując zakodowaną reprezentację ścieżki z innymi zakodowanymi
reprezentacjami jesteśmy w stanie znaleźć utwory mające podobne warotści atrybutów
w przestrzeni ukrytej.


### Model 2 

Opis modelu znajduje się w pliku .


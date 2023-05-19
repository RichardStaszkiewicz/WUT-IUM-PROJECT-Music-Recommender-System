# IUM 
## System rekomendacji utworów

### Poniższy raport przedstawia:
- opis wykorzystywanych modeli
- trening modeli
- porównanie wyników modeli
- implementację mikroserwisu
- testy A/B

### Wykorzystane dane

W rozwiązaniu wykorzystano zbiory danych:
- tracks
- sessions
- users

Dodatkowo modele są trenowane na danych do dnia `2022-09-01`. Rekomendacje są tworzone na bazie ostatniego miesiąca sesji danego użytkownika.
Modele są walidowane bazując na rekomendacjach i na kolejnym miesiącu odsłuchań użytkownika, tj. `2022-09-01`.

<img src="plots/barplot_sessions_time.jpg" alt="Loss model" width="700">


Dodatkowo zbiór `sessions` został ograniczony do użytkowników o id: `141, 167, 200, 206, 237, 281, 287, 297, 301, 324, 362, 386, 394, 418, 429, 433, 434, 442, 486, 505`
dane sesji użytkowników. Zdecydowano się na taki krok, ponieważ filtrowanie odpowiednich rekordów po danym czasie i użytkowniku zajmuje znaczą część czasu, obniżając wydajność systemu.
Klient będzie zobowiązany do zaimplementowania wydajnej bazy danych i zapytań `.sql` zwracających rekordy sesji. Przygotowano odpowiednie skrypty na integrujące zewnętrze api klienta do baz danych.

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

Podczas treningu osiągnięto zadowalający błąd między rzeczywistymi danymi a rekonstrukcjami.

![Loss model](plots/loss_model_1.png)

### Model 2 

Opis modelu znajduje się w pliku.

### Uruchomienie

Poniższa instrukcja ukazuje jak krok po korku wykorzystywać serwis:

1. Przejść do katalogu `microservice` oraz zainstalować potrzebne pakiety `python` za pomocą polecenia:

`
pip install -m requirements.txt
`

2. uruchomić skrypt `./run.sh`
Uruchomi on mikroserwis pod lokalnym adresie: `127.0.0.1:8000`

Należy odczekać kilka sekund, aż serwis będzie aktywny. Poniższe komunikaty wskazują na zakończenie startu serwisu.

```
INFO:     Started server process [24360]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

3. Uruchomić aplikację [Postman](https://www.postman.com/downloads/) i załadować plik konfiguracji: `postman_collections.json`.
4. Korzystając z wyświetlających się endpointów można korzystać z serwisu

![Postman endpoints](plots/postman_endpoints.png)

### Opis Endpointów

Plik postman_collection.json to kolekcja Postmana zawierająca wszystkie endpointy.


#### `POST /perform_ab_test`

Służy do zbierania danych do testów A/B. Zwraca czy rekomendacje okazały się poprawne czy nie, dla danego użytkownika.

```
{
    "user_id": 429
}
```
Przykładowa odpowiedź: `{"Successful recommendation": 1}`

#### `DELETE /ab_test/results`

Czyści plik przechowujący wyniki testów A/B.

#### `GET /ab_test/results`

Zwraca zawartość pliku przechowujący wyniki testów A/B.

#### `POST /models/{model_id}/predict`

Zwraca predykcję dla danego modelu.

- model_id = 1: Enkoder 
- model_id = 2: Klasyfikator

Dane wejściowe i wyjściowe są w takim samym formacie.

Zwraca listę zawierającą `track_id` rekomendowanych piosenek.

Przykładowa odpowiedź:

```
[
    "0ofHAoxe9vBkTCp2UQIavz",
    "6zFMeegAMYQo0mt8rXtrli",
    "5vGLcdRuSbUhD8ScwsGSdA",
    "4xqrdfXkTW4T0RauPLv3WA",
    "1KixkQVDUHggZMU9dUobgm",
    "54bFM56PmE4YLRnqpW6Tha",
    "1tkg4EHVoqnhR6iFEXb60y",
    "7vrJn5hDSXRmdXoR30KgF1",
    "45bE4HXI0AwGZXfZtMp8JR",
    "35mvY5S1H3J2QZyna3TFe0",
    "3YJJjQPAbDT7mGpX3WtQ9A",
    "3FAJ6O0NOHQV8Mc5Ri6ENp",
    "6f3Slt0GbA2bPZlz0aIFXN",
    "0VjIjW4GlUZAMYd2vXMi3b",
    "6tDDoYIxWvMLTdKpjFkc1B",
    "5QO79kh1waicV47BqGRL3g",
    "2gMXnyrvIjhVBUZwvLZDMP"
]
```




 







# Klasyfikacja ruchu sieciowego: Bot vs Człowiek

## Opis projektu

Projekt służy do analizy i klasyfikacji ruchu sieciowego w celu rozróżnienia ruchu generowanego przez boty od ruchu generowanego przez ludzi. Wykorzystuje on pliki PCAP z zarejestrowanym ruchem, z których ekstraktowane są cechy w oknach czasowych (1s i 5s), a następnie trenowane i testowane są różne modele klasyfikacji.

## Główne funkcjonalności

1.  **Generowanie ruchu botów**  
    W katalogu znajdują się skrypty (`bot1_scrapy_crawler.py`, `bot2_selenium_browser.py`, itd.) generujące różne typy ruchu botów (np. przeglądanie stron, pobieranie plików, wypełnianie formularzy). Skrypt `run_all_bots.py` pozwala uruchomić wszystkie boty sekwencyjnie.

2.  **Przetwarzanie plików PCAP i ekstrakcja cech**  
    Skrypty `process_pcap_1s.py` i `process_pcap_5s.py` analizują pliki PCAP (`human_traffic.pcapng`, `bot_traffic.pcapng`) i dzielą ruch na okna czasowe (1s lub 5s). Dla każdego okna wyliczane są cechy statystyczne, takie jak liczba pakietów, bajtów, statystyki rozmiarów pakietów, czasy między pakietami, stosunki wysłanych/odebranych pakietów i bajtów, itp. Wyniki zapisywane są do plików CSV (`labled_trafic_1s.csv`, `labled_trafic_5s.csv`).

3.  **Trenowanie i ewaluacja modeli klasyfikacyjnych**  
    Skrypty `train_evaluate.py`, `train_evaluate_1s.py`, `train_evaluate_5s.py` służą do trenowania i testowania modeli klasyfikacyjnych (KNN, SVM, Random Forest, Voting, Stacking) na wyekstrahowanych cechach. Wyniki ewaluacji (accuracy, precision, recall, F1-score) zapisywane są w pliku `classification_reports.txt`.

4.  **Wizualizacja wyników**  
    Skrypt `visualize_results.py` umożliwia generowanie wykresów i tabel podsumowujących skuteczność modeli oraz zapisuje je w katalogu `result_plots/`.

## Struktura katalogów i plików

-   `bot*_*.py` – skrypty generujące ruch botów
-   `run_all_bots.py` – uruchamianie wszystkich botów
-   `human_traffic.pcapng`, `bot_traffic.pcapng` – pliki z ruchem sieciowym
-   `process_pcap_1s.py`, `process_pcap_5s.py` – ekstrakcja cech z PCAP
-   `labled_trafic_1s.csv`, `labled_trafic_5s.csv` – cechy z okien czasowych
-   `train_evaluate.py` – główny skrypt do trenowania i ewaluacji (dla danych z `labled_trafic.csv` - domyślnie 5s)
-   `train_evaluate_1s.py` – skrypt do trenowania i ewaluacji dla danych 1-sekundowych
-   `train_evaluate_5s.py` – skrypt do trenowania i ewaluacji dla danych 5-sekundowych
-   `classification_reports.txt` – szczegółowe raporty klasyfikacji
-   `visualize_results.py` – wizualizacja wyników
-   `result_plots/` – katalog z wykresami

## Wymagania

-   Python 3.8+
-   Pakiety: scikit-learn, pandas, numpy, matplotlib, seaborn, scapy, scrapy, selenium, webdriver-manager

Przykładowa instalacja (możesz stworzyć plik `requirements.txt` z listą pakietów):
```bash
pip install scikit-learn pandas numpy matplotlib seaborn scapy scrapy selenium webdriver-manager
```

## Sposób użycia

1.  **Przygotowanie środowiska i danych**
    *   Upewnij się, że masz zainstalowane wszystkie wymagane pakiety.
    *   Umieść pliki PCAP z ruchem ludzkim (`human_traffic.pcapng`) i botów (`bot_traffic.pcapng`) w głównym katalogu projektu. Możesz też wygenerować ruch botów za pomocą dostarczonych skryptów (`bot*_*.py` i `run_all_bots.py`).

2.  **Ekstrakcja cech z plików PCAP**  
    Uruchom skrypty, aby przetworzyć pliki PCAP i wygenerować pliki CSV z cechami:
    ```bash
    python process_pcap_1s.py
    python process_pcap_5s.py
    ```
    *Uwaga: Skrypt `process_pcap_5s.py` domyślnie zapisuje wyniki do `labled_trafic.csv`, a `process_pcap_1s.py` do `labled_trafic_1s.csv`. Skrypt `train_evaluate.py` używa `labled_trafic.csv`.*

3.  **Trenowanie i ewaluacja modeli**  
    Możesz uruchomić główny skrypt, który użyje domyślnego pliku `labled_trafic.csv` (wygenerowanego przez `process_pcap_5s.py` jeśli jego `OUTPUT_CSV` to `labled_trafic.csv`):
    ```bash
    python train_evaluate.py 
    ```
    Alternatywnie, możesz uruchomić skrypty dla konkretnych długości okien (upewnij się, że odpowiednie pliki `INPUT_CSV` są skonfigurowane w tych skryptach lub zmień ich nazwy na domyślne):
    ```bash
    python train_evaluate_1s.py  # Używa labled_trafic_1s.csv
    python train_evaluate_5s.py  # Używa labled_trafic_5s.csv
    ```

4.  **Wizualizacja i generowanie raportów**  
    Po trenowaniu modeli, skrypt `visualize_results.py` może zostać użyty do wygenerowania podsumowujących wykresów i raportów. Ten skrypt również trenuje modele na nowo, więc upewnij się, że pliki CSV (`labled_trafic_5s.csv`, `labled_trafic_1s.csv`) są dostępne.
    ```bash
    python visualize_results.py
    ```
    Spowoduje to również wygenerowanie/aktualizację pliku `classification_reports.txt` oraz wykresów w katalogu `result_plots/`.

## Wyniki

Wyniki klasyfikacji (accuracy, precision, recall, F1-score) dla różnych modeli i okien czasowych znajdują się w pliku `classification_reports.txt`. Dla okien 5s modele osiągają bardzo wysoką skuteczność. Dla okien 1s skuteczność jest również wysoka, choć nieco niższa, szczególnie w przypadku klasyfikacji okien pustych (gdy nie ma ruchu).

## Autor

Mikele-Kochas

## Repozytorium

[https://github.com/Mikele-Kochas/Klasyfikacja-ruchu-bot-vs-cz-owiek](https://github.com/Mikele-Kochas/Klasyfikacja-ruchu-bot-vs-cz-owiek) 
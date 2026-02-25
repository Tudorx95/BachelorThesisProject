# Functionalitati principale implementate in proiectul de diploma

## Functionalitati platforma

- Procedee de autentificare si inregistrare utilizator 

- Creare de proiecte cu diferite scenarii de testare

- Crearea de teste diverse in proiecte, precum si ordonarea lor din interfata, atat intre fisiere de test, cat si intre proiecte

- Editor de text performant (syntax highlighting) cu un template tensorflow predefinit pentru editare + buton de copy template in clipboard 

- Platforma suporta doua template-uri de cod cu suport pt tensorflow si pytorch

- Simularea unui mediu federated learning folosind urmatoarele tipuri de atacuri data poisoning:


    - label-flip (modificarea clasei din care fac parte un esantion de imagini cu o clasa tinta aleasa de utilizator sau random)
    
    - backdoor badnets  (adauga un trigger mic in forma de cruce, sau alte forme, pe un anumit procent din imagine)
    
    - backdoor blended injection attack ( un tip de atac ce presupune generarea unei noi imagini prin combinarea imaginii de baza cu un key pattern. Acest key pattern poate fi si un sir random de numere, asa cum mentioneaza si paperul:  https://arxiv.org/pdf/1712.05526)
    
    - backdoor sinusoidal (adaugarea in imagine a unui semnal sinusoidal a carui parametrii sunt setati de utilizatorul platformei mele, precum frecventa si amplitudinea)
    
    - backdoor trojan (un tip de atac in care se insereaza in imagine un watermark; este similar cu backdoor badnets)
    
    - backdoor semantic (atac ce presupune modificarea unor caracterisitici naturale ale imaginii: luminozitate, nuanta RGB)
    
    - edge case backdoor (alte modalitati de atac precum modificarea unghiului imaginii, reducerea numarului de culori, solarizare/crestere in intensitate luminoasa, reducere intensitate spre negru/tonuri de gri).

- In cadrul mediului federated learning, reteaua e formata din N clienti dintre care M sunt malitiosi. Antrenarea va consta in rularea a ROUNDS runde, timp in care clientii malitiosi vor folosi datele poisoned R runde din ROUNDS.

- Distribuirea caracterului de actor malitios intre clienti: 

    - first - primii M din N 
    - last - ultimii M din N
    - alternate - primul malitios, al doilea bun, etc. 

- Mecanisme de protectie impotriva atacurilor data poisoning (metode robuste de agregare a ponderilor):

    - Federated Averaging (FedAvg) — media ponderata clasica a ponderilor clientilor, proportionala cu dimensiunea dataset-ului local. Este metoda standard, dar vulnerabila la atacuri de data poisoning.

    - Krum — selecteaza update-ul unui singur client care are cea mai mica suma a distantelor fata de ceilalti clienti. Elimina eficient ~99% din atacuri prin izolarea clientilor malitiosi.
    
    - Trimmed Mean — elimina un procent (20%) din cele mai extreme valori (sus si jos) ale ponderilor si calculeaza media pe valorile ramase. Rezistent in special la atacuri de tip label-flipping.
    
    - Median — inlocuieste media cu mediana ponderilor per fiecare parametru al retelei neuronale. Rezistent la pana la 20% clienti malitiosi.
    
    - Trimmed Mean + Krum — abordare hibrida ce combina filtrarea extremelor prin Trimmed Mean (trim_ratio=0.1) cu selectia Krum.
    
    - Random — selecteaza aleatoriu ponderile unui singur client. Folosit ca baseline de comparatie.

- Simularea ruleaza automat 3 scenarii pentru fiecare test:
    
    1. **Clean** — antrenare fara clienti malitiosi (baseline)
    
    2. **Poisoned** — antrenare cu clienti malitiosi folosind FedAvg (fara protectie)
    
    3. **Poisoned + Data Protection** — antrenare cu clienti malitiosi folosind metoda de protectie selectata

- Metrici de evaluare colectate per runda si per scenariu:
    - Accuracy (acuratete globala)
    - Precision (media ponderata — weighted average)
    - Recall (media ponderata — weighted average)
    - F1 Score (media ponderata — weighted average)

## Functionalitati frontend

- Interfata tip notebook (inspirata Jupyter) cu celule de cod si celule de output

- Sidebar cu gestiunea proiectelor si fisierelor de test, cu posibilitatea de reordonare prin drag-and-drop

- Vizualizarea progresului simularii in timp real, cu status per etapa (clean, poisoned, poisoned+DP) prin componenta ProgressStep

- Afisarea rezultatelor finale (summary text + analiza JSON detaliata) la completarea simularii

- Export rezultate individuale in format PDF (include configuratie, metrici de acuratete si confusion matrix, summary)

- Export rezultate multiple in format CSV (selectie multipla de fisiere de test, coloane pentru toate metricile per scenariu)

- Mod dark / light cu persistenta preferintei utilizatorului

- Configurare avansata a parametrilor simularii prin modal dedicat (SimulationOptions): numar clienti, numar malitiosi, runde, strategie de distributie, tip de atac, parametri specifici fiecarui atac, metoda de protectie

- Pagina **Compare** — permite compararea detaliata a doua simulari selectate din acelasi proiect. Afiseaza side-by-side: configuratia FL (N, M, ROUNDS, R, strategie), parametrii atacului de data poisoning (tip operatie, intensitate, procent date afectate), si rezultatele simularii (Init/Clean/Poisoned/Poisoned+DP Accuracy, accuracy drop, metoda de protectie, GPU utilizat). Include si un summary text complet per simulare.

- Pagina **Graphs** — vizualizare grafica a rezultatelor mai multor simulari prin grafice bar chart interactive (folosind Recharts). Utilizatorul poate selecta una sau mai multe simulari din proiectul activ, iar graficul grupeaza metricile pe categorii (Init Accuracy, Clean Accuracy, Poisoned Accuracy, Poisoned + DP Protection) cu bare colorate per simulare. Include si un tabel detaliat cu valorile exacte ale metricilor pentru fiecare simulare selectata.

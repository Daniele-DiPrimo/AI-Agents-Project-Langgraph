# Teorema del Campionamento

## G. MARSELLA
## UNIVERSITÀ DEL SALENTO

---

## Il processo di conversione A/D

Il processo di conversione A/D comporta la trasformazione di un segnale continuo (analogico) in un insieme finito di valori (discretizzazione).

Poiché un segnale analogico è continuo sia nel TEMPO che in AMPIEZZA sono necessarie due fasi:

*   CAMPIOΝΑΜΕΝΤΟ
    *   discretizzazione nel TEMPO
*   QUANTIZZAZIONE
    *   discretizzazione in AMPIEZZA

---

## Il teorema di Shannon

Quanti campioni/s sono necessari per campionare adeguatamente un segnale? ovvero

Qual è la frequenza di campionamento minima che mi permette di ricostruire il segnale in modo univoco dai suoi campioni?

Devo prendere almeno 2 campioni per ogni periodo

Per campionare un segnale sinusoidale di 1 Hz devo prendere al minimo $f_c = 2$ Hz

[Grafico: Andamento di un segnale sinusoidale con indicazione delle frequenze di campionamento $f_c = 1$ Hz e $f_c = 2$ Hz]

---

## Dati analogici, segnali numerici

Per poter trasmettere un dato analogico con una trasmissione digitale e' necessario trasformare il dato analogico in un segnale numerico. Più precisamente si rappresenta il segnale analogico, corrispondente al dato analogico in banda base, con un dato numerico.

Il processo di trasformazione si realizza attraverso due fasi:

*   il campionamento del segnale analogico
*   la digitalizzazione del campione

---

## Il campionamento

Il campionamento consiste nel guardare con una certa frequenza il valore istantaneo del segnale analogico. Di fatto si utilizza il segnale analogico per modulare in ampiezza una sequenza di impulsi a frequenza fissata: il segnale risultante sarà una sequenza di impulsi ad ampiezza uguale al valore del segnale analogico in corrispondenza degli impulsi.

Il problema da affrontare è: con quale frequenza si deve campionare il segnale per poterlo ricostruire a partire dal segnale campionato?

[Grafico: Segnale analogico x(t) campionato a intervalli T, 2T, 3T]
[Grafico: Segnale campionato x_s(t) come sequenza di impulsi modulati in ampiezza]

---

## Teorema del campionamento

IL teorema del campionamento (o teorema di Nyquist-Shannon) afferma che:

dato un segnale $x(t)$ il cui spettro ha banda limitata $B$, si può ricostruire completamente il segnale a partire da un campionamento dello stesso se la frequenza di campionamento è $F \geq 2B$.

---

## Dimostrazione

Sia $x(t)$ il segnale di banda $f_h$.
Sia $p(t)$ il segnale di campionamento a frequenza $f_s$.
Il segnale campionato sarà:
$x_s(t) = x(t) \cdot p(t)$

dove $p(t) = \sum_{n=-\infty}^{\infty} P_n e^{i 2 \pi n f_s t}$

La trasformata del segnale campionato è:
$X_s(f) = \int_{-\infty}^{\infty} x_s(t) e^{-i 2 \pi f t} dt = \int_{-\infty}^{\infty} \sum_{n=-\infty}^{\infty} P_n x(t) e^{i 2 \pi n f_s t} e^{-i 2 \pi f t} dt$

Quindi:
$X_s(f) = \sum_{n=-\infty}^{\infty} P_n \int_{-\infty}^{\infty} x(t) e^{-i 2 \pi (f - n f_s) t} dt$

---

## Dimostrazione (cont.)

$X_s(f) = \sum_{n=-\infty}^{\infty} P_n \int_{-\infty}^{\infty} x(t) e^{-i 2 \pi (f - n f_s) t} dt$

La trasformata del segnale è:
$X(f) = \int_{-\infty}^{\infty} x(t) e^{-i 2 \pi f t} dt$

da cui:
$X_s(f) = \sum_{n=-\infty}^{\infty} P_n X(f - n f_s)$

Questo significa che lo spettro del segnale campionato è costituito da repliche dello spettro del segnale originale traslate ai multipli della frequenza del segnale di impulsi utilizzato per campionarlo, e moltiplicate ciascuna per un fattore proporzionale ($P_n$).

---

## Dimostrazione (cont.)

Se gli spettri di due repliche adiacenti del segnale originario non si sovrappongono, possiamo utilizzare in ricezione un filtro passa basso per isolare una sola replica del segnale, ottenendo così un segnale il cui spettro è proporzionale (cioè ha forma identica) allo spettro del segnale originale.

La condizione di non sovrapposizione implica:
$f_h \leq f_s - f_h \implies f_s \geq 2 f_h$

cioè quello che si voleva dimostrare.

[Grafico: Spettro del segnale originale x(t)]
[Grafico: Spettro del segnale campionato x_s(t) con repliche sovrapposte se $f_s < 2f_h$ e non sovrapposte se $f_s \geq 2f_h$]

---

## Osservazioni sul teorema del campionamento

In pratica la frequenza di campionamento dovrà essere almeno leggermente superiore a $2B$, per disporre di un intervallo utile (banda di guardia) al fine di prevenire che effetti di non idealità dei filtri taglino parti utili del segnale.

Il teorema del campionamento è sostanzialmente collegato alla legge sulla massima capacità di un canale privo di rumore (legge di Nyquist):

*   il teorema del campionamento afferma che possiamo ricostruire il segnale campionando almeno a $2B$, e campionando più frequentemente non otteniamo maggiori informazioni sul segnale modulante.
*   se il segnale rappresenta una sequenza di simboli, la massima capacità di trasferimento la otteniamo quando ogni campione identifica un simbolo.
*   ne segue che al massimo siamo in grado di identificare $2B$ simboli.

---

## Teorema del campionamento

Il risultato del campionamento è un segnale con valori discreti. Tale segnale sarà in seguito quantizzato e codificato per renderlo accessibile a qualsiasi elaboratore digitale.

Il teorema del campionamento pone un vincolo per la progettazione di apparati di conversione analogico-digitale: se si ha a disposizione un campionatore che lavora a frequenza $F_c$, è necessario mandargli in ingresso un segnale a banda limitata da $F_c/2$ (Teorema di Shannon).

In generale un segnale analogico non è limitato in frequenza, ma dovrà essere filtrato per eliminare le componenti di frequenza maggiore di $F_c/2$, a tale scopo si usa un filtro anti-aliasing (filtro passa-basso).

Un segnale $f(t)$ a banda limitata da $f_M$ (frequenza massima) può essere univocamente ricostruito dai suoi campioni $f(n \Delta t)$ con $n \in \mathbb{N}$ presi ad una frequenza $F_c = 1/\Delta t$ solo se $F_c \geq 2 f_M$.

Es. l'orecchio umano è in grado di percepire frequenze tra i 20Hz e i 22 KHz; la frequenza di campionamento per l'audio (limite di Nyquist) si pone quindi attorno ai 44 KHz.

---

## Effetto Aliasing

Consiste in una sovrapposizione del segnale campionato che rende impossibile l'esatta ricostruzione del segnale originale e tale ricostruzione risulterà distorta.

Per questo motivo ogni apparato di conversione analogico-digitale ha un filtro anti-alias (filtro passa basso) a monte del campionatore, che limita lo spettro del segnale di ingresso a $F_c \geq 2 f_M$.

---

## Il teorema di Shannon-Nyquist e la frequenza di campionamento

Dato un segnale continuo e a banda limitata esso è descritto completamente dai suoi campioni, se essi sono presi ad una frequenza almeno doppia rispetto alla frequenza massima del segnale.

Es: Periodo $T = 37$ms $\implies f_{segnale} = 1/(37 \text{ ms}) \approx 27$ Hz

[Grafico: Andamento di un segnale sinusoidale con indicazione dell'area del tracciato]
[Diagramma a blocchi: Segnale in ingresso -> ADC (8 bit, clock, Sample Clock) -> DAC -> Uscita (?)]
Indicazione delle frequenze: $f_{campionamento}$

---

## Spettro di un segnale

Il campionamento introduce un aliasing: in pratica il campionamento provoca "duplicazioni" dello spettro del segnale. Se non si rispetta il teorema di Shannon l'aliasing introdotto dal campionamento impedisce la ricostruzione del segnale originale, in quanto due spettri adiacenti si sovrappongono!

[Diagramma a blocchi: Segnale in ingresso -> ADC]
[Grafico: Spettro del segnale in ingresso con banda $f_s$]
[Grafico: Spettro del segnale campionato con $f_c > 2 f_s$ e bande sovrapposte]

---

## Spettro di un segnale

L'effetto di un errato campionamento, nell'ambito del dominio delle frequenze:

Per un buon funzionamento del campionatore e per evitare aliasing, si introduce un filtro passa basso che limita lo spettro del segnale di ingresso a $f_c > 2 f_s$.

[Grafico: Spettro del segnale campionato con $f_c < 2 f_s$ e bande sovrapposte]
[Diagramma a blocchi: Filtro PB -> ADC]

---

## Tecniche di modulazione di treno di impulsi

Esistono diverse tecniche di modulazione:

*   PAM (Pulse Amplitude Modulation): gli impulsi sono generati ad ampiezza proporzionale alla ampiezza del segnale modulante.
*   PWM (Pulse Width Modulation): gli impulsi sono generati tutti alla stessa ampiezza, ma con durata proporzionale alla ampiezza del segnale modulante.
*   PPM (Pulse Position Modulation): gli impulsi sono tutti della stessa ampiezza e di uguale durata, ma iniziano (all'interno del periodo T) in un istante dipendente dalla ampiezza del segnale modulante.
    *   in questo caso il ricevente deve essere sincronizzato con il trasmittente in quanto la valutazione dell'ampiezza del segnale modulante dipende dalla differenza temporale tra l'istante in cui si presenta l'impulso e l'istante in cui inizia il periodo relativo a quell'impulso, quindi in ricezione si deve sapere quando inizia il periodo relativo all'impulso.

---

## PWM e PPM

[Illustrazione comparativa di segnale analogico modulante, treno di impulsi portante, segnale modulato PPM e segnale modulato PWM.]

---

## Considerazioni sullo spettro

La trasmissione di un treno di impulsi di durata $\tau$ richiede una larghezza di banda almeno pari a:
$B_\tau \geq \frac{1}{2\pi\tau}$

ed essendo $\tau \ll T$ e $T \leq \frac{1}{2B}$ si ha:
$B_\tau \geq \frac{1}{2\tau} \gg B$

significa che la trasmissione di impulsi modulati richiede una banda superiore alla banda del segnale modulante.

---

## Trasmissione radio/TV

L'esempio più comune di FDM è la trasmissione radiotelevisiva. Questa utilizza diverse bande di frequenza, ciascuna delle quali viene suddivisa in canali di una certa capacità, idonea a trasmettere i segnali delle diverse stazioni trasmittenti.

*   trasmissioni a modulazione di ampiezza (AM) nella banda MF (Medium Frequency): 300-3000 KHz, con canali da 4 KHz per radio commerciali.
*   trasmissioni AM nella banda HF (High Frequency): 3-30 MHz, con canali fino a 4 KHz (radio onde corte).
*   trasmissioni AM o FM nella banda VHF (Very High Frequency): 30-300 MHz, con canali fino a 5 MHz (radio FM e TV VHF).
*   trasmissioni FM nella banda UHF: 300-3000 MHz con canali fino a 20 MHz (TV UHF, ponti radio).
*   trasmissioni FM nella banda SHF: 3-30 GHz con canali fino a 500 MHz (microonde terrestri e satellitari).

---

## ADSL

ADSL (Asymmetric Digital Subscriber Line) è lo standard per fornire all'abbonato un accesso digitale a banda più elevata di quanto non sia possibile con il modem.

La linea telefonica terminale è costituita da un doppino su cui viene normalmente trasmessa la voce. Questa trasmissione si realizza applicando un filtro passa basso a 4 KHz.

Tuttavia il doppino ha una capacità di banda che raggiunge il MHz (dipende dalla lunghezza del tratto terminale che può variare tra poche centinaia di metri a diversi Km).

Lo spettro disponibile viene suddiviso in 256 canali da 4 KHz (fino a 60 Kbps ciascuno):

*   Il canale 0 viene riservato per la telefonia.
*   I successivi 4 canali non vengono utilizzati per evitare problemi di interferenza tra la trasmissione dati e quella telefonica.
*   I restanti canali vengono destinati al traffico dati. Alcuni per il traffico uscente (upstream), altri per il traffico entrante (downstream).

Il modem ADSL riceve i dati da trasmettere e li separa in flussi paralleli da trasmettere sui diversi canali, genera un segnale analogico in banda base per ciascun flusso (con una modulazione QAM fino a 15 bit/baud a 4000 baud/s) e li trasmette sui diversi canali utilizzando la modulazione di frequenza.
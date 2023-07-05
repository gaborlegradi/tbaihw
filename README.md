# Házi feladat a Turbine részére - 1st part

## 2023 június 15 utáni megjegyzések

### 2023 június 5-i összefoglaló

Az eredeti koncepcióm az volt, hogy
1. betanítok egy Variational AutoEncodert (VAE),
2. majd ennek Encoder részére ráteszek egy classifier head-et és azt úgy tanítom be, hogy az Encoder esetében trainable=False beállítással fagyasztom a taníthatóságot.

Ezt először a MNIST adatokon (kézzel írt számjegyek) próbáltam ki. Mivel a számjegyek esetében minden pixel (ezek képezik az input csatornákat) releváns információt hordoz, ezért nagyon jól rekonstruálhatónak bizonyultak, és a koncepcióm jól működött.

#### Variational AutoEncoderek 

A VAE-k eredetileg generatív eszköznek lettek kitalálva főleg képek generálásához, a GAN-ok mellett a legnépszerűbbek voltak erre a célra egy időben. VAE úgy jön létre, hogy egybeépítünk egy olyan Encoder + Decoder párost, 
1. amelynek összeillesztésénél a neckben, a reprezentáció nagyon kis dimenziószámra (latent_dim) van szorítva a bemeneti csatornaszűmhoz képest, illetve
2. a neckben az Encoder kimenete nem fix érték, hanem valamilyen eloszlás, jelen esetben Gauss-eloszlás, amit a tanítás során egyszerű Monte-Carlo módszerrel mintavételezünk, és
3. a lossban a rekonstrukciós loss mellett megjelenik a KL divergencia, ahol a KL divergencia referencia eloszlása a normál-eloszlás ($\mu=0$, $\sigma=1$).

Egy [kifejezetten jó leírás található itt](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) a VAE-kről, amely megmutatja a koncepcó mögött meghúzódó Bayesiánus gondolatot is.

A VAE-k azt célozzák, hogy a tanítóadatot tipizálják, és ezt a neck-ben úgy reprezentálják, hogy a reprezentáció **teljes** és **folytonos**. A teljesség alatt azt értjük, hogy a tanítóadat mindegyikét reprezentálja a latent_dim-beli reprezentáció valmely területe, illetve a latent_dim-beli reprezentáció minden pontja - egy központi területen - "értelmes" adatot reprezentál, ezt a pontot a Decoder-re beadva éretlmezhető rekonstrukciót kapunk - pl számjegyet, ha számjegyekkkel tanítottunk, vagy emberi arcokat, ha emberi arcokkal tanítottunk. Ezt a reconstrukciós loss (pl L2 az input-outputra) és a KL divergencia együttes használatával érjük el.

A netutils.py-ban van definiálva a Sampling osztály, ez valósítja meg a neckben a latent_dim számú neuronon a reprezentációt, és a create_encoder függvényben ugyanitt látható, hogy hogyan épül be az Encoder-be. A Sampling layer minden inputra egy várható értéket (z_mean) és szigmát (z_log_var, egész pontosan ez a variancia e alapú logaritmusának kétszerese) ad vissza, majd ezt mintavételezi és így kapjuk a z-t, ami egy latent_dim hosszúságú vektor. A KL divergencia két _p_($\mu_1$, $\sigma_1$) és _q_($\mu_2$, $\sigma_2$) Gauss-eloszlás esetén, ahol _q_ a referencia-eloszlás: 

$$ KL(p,q) = log \dfrac{\sigma_2}{\sigma_1} + \dfrac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \dfrac{1}{2} $$

Amennyiben a referencia eloszlás N(0, 1), akkor

$$ KL(p,N(0, 1)) = -log(\sigma_1) + \dfrac{\sigma_1^2 + \mu_1^2}{2} - \dfrac{1}{2} $$

Ez a loss tag jelenik meg a vaec_trainer.py-ban a calc_loss függvényben a 76-77-ik sorban.


### 2023 június 23-i megjegyzések

A megoldásom elküldésekor is már jeleztem, hogy nem vagyok vele elégedett. Bár az elmúlt időszakban nem tudtam vele gép előtt foglalkozni, végig a fejemben volt és úgy gondolom, hogy most már van egy olyan megoldáskoncepcióm, amivel minden kérdésre megfelelő választ tudok adni.

Amikor a feladatnak nekiálltam, először a MNIST adatokon (kézzel írt számjegyek) próbáltam ki. Mivel a számjegyek nagyon jól rekonstruálhatónak bizonyultak, a koncepcióm jól működött, de ez a pozitív tapasztalat félrevezetett. A Wisconsin Breast Cancer Database esetében ugyanis a bemenet nem jól rekonstruálható, pontosan amiatt, hogy a bemeneti adat számos csatornáján irreleváns információ áll rendelkezésre - aminek lehetőségét én magam is felvetettem, illetve az adatleírás is utalt erre. Ezért a VAE tanítás nem eredményezhet olyan autoencodert, ami jó performansszal rekonstruálhatna; a VAE tanítás során "lehetetlen feladatot" akarunk megoldani, a rekonstrukciós loss a tanítás során csak "rázza" a tanítandó **w<sub>i</sub>**, **b<sub>i</sub>** értékeket. Emiatt, a VAE tanítással nem kapunk olyan encodert, ami a számunkra érdekes információt desztillálná. A helyes megoldás koncepcióját az alábbiakban összegzem:

-  Bár pont ezt a kombinációt eredetileg nem teszi lehetőve a VAEC_Trainer (vaec_trainer.py), pontosan a variational réteget (Sampling, netutils.py) tartalmazó encodert és a rá ültetett mlp_classifier_headot kell egybe tanítani. Így egy Variational MLP Classifier (VMLPC) jön létre.
-  A VMLPC-t be kell optimalizálni oly módon, hogy a következő elvárásoknak egyszerre felelünk meg:
  - A **latent_dim** a lehető legkisebb legyen oly módon, hogy
  - mindeközben a precision és recall értékekkel minősített performancera vonatkozóan a performance cost legyen minimális egy optimalizált full-MLP performanszához képest, illetve
  - a tanító adatot a Sampling rétegben a **z** vektorokkal reprezentálva elvárjuk, hogy a **z** tagjai (**z<sub>i</sub>**) is és **|z|** is normál eloszlásúak legyenek.

Ez utóbbi feltétellel biztosítjuk és bizonyítjuk, hogy a Sampling réteg és a KL Divergence loss alkalmazása elérte a kívánt hatást, vagyis a tanítóadat reprezentációját ráfeszítette a normál eloszlásra.

Ezt követően a Task6-ra és Task7-re a következő válaszokat tudom adni:

## Task6
A Task 6-nál a válaszom ezúttal az, hogy amennyiben szignifikáns dimenzió redukciót érek el az encoderrel, akkor vagy arról van szó, hogy 
- vannak irreleváns paraméterek,
- vagy a modell enkóder része "érdekes" összefüggések megtanulása révén tudta a dimenzió redukciót ilyen - remélhetőleg - markáns mértékben megoldani.

Az irreleváns paramétereket oly módon állapíthatjuk meg, hogy megvizsgáljuk, hogy egy VAE tanítás során mely csatornákon nem bizonyul lehetségesnek a rekonstrukció. (A **latent_dim**-et már előtte, a fent írtak szerint meghatároztuk.) Mivel a VAE tanítás nem konvergál, itt lehet, hogy egy többlépcsős algoritmust kell alkalmazni: Az első tanítási próbálkozás, vagy több próbálkozás eredményeképpen meg kell állapítani, hogy mely input csatornák a legkevésbé rekonstruálhatóak. Ezeket ki kell venni a lossból, majd ezt követően ismét tanítani kell azzal a céllal, hogy a megmaradt csatornák közül kiválasszuk a legkevésbé rekonstruálhatóakat. Ezt a külső iterációt addig kell folytatni, amíg a bemeneten meghagyott csatornák mindegyike jól rekonstruálható lesz. A többit tekinthetjük irreleváns információ hordozójának.

## Task7
A Task7-re adott válaszom lényegében nem változott, de kis mértékben finomítottam:

Egy adott input prediktálása során a latent_dim méretű **z** output vektor minden egyes **z<sub>i</sub>** elemére T-teszttel megmondjuk, hogy milyen konfidencia érték mellett tekinthető az értéke egy normál-eloszlásból történő mintavételezésnek. Így egy konfidencia-vektort kapunk. Ugyanezt a tesztet elvégezhetjük a vektor **|z|** hosszértékével is, ekkor egyetlen konfidencia értéket kapunk. Ezzel tehát azt becsülnénk meg inputonként (egy adatsor 30 bemenő értékkel), hogy milyen konfidenciával jelenthetjük ki, hogy legalábbis a neck-beli reprezentációs térben az eredeti tanító adatnak megfelelő normál eloszlás mintavéltelének tekinthető-e az adott input? Amennyiben csak a **z<sub>i</sub>** bizonyos értékei lógnak ki, lehetséges, hogy meg tudjuk mondani, hogy milyen feature tekintetében vagyunk eloszláson kívüliek. Ennek feltétele, hogy a latent_dim beli reprezentációnál tudunk-e szemantikai értelmezést adni a dimenzióknak.


## A 2023 június 15-én beküldött megoldás

A feladat taskjain végighaladva foglalom össze a munkámat.

## A választott megoldás összefoglalása + Task1
A házi feladat első részének kiírásában a Task6 és Task7 mozgatta meg legjobban a fantáziámat. Úgy ítéltem meg, hogy ezen kérdésekre akkor tudok válaszolni, ha a modellemnek része egy variational autoencoder. Az alapötletem az volt, hogy betanítok egy VAE-t, majd ennek az encoder részére ráhelyezve egy MLP classifier head-et kapok egy tisztán MLP classifiert.

Egy megoldást gyorsan össze is drótoztam és teszteltem MNIST adaton, ezzel az adattal ugyanis nagyon jól lehet érzékeltetni latent_dim=2-re a reprezentációt a VAE neck-jében. Itt már a latent_dim = 1 esetre is ~70% körüli accuracy-t kaptam, a latent_dim 2, illetve 3 értékénél 85% és 95% fölötti accuracy tűnt elérhetőenk. Ezt ígéretesnek ítélve rátértem a Wisconsin Breast Cancer Dataset-re készítendő megoldásomra.

A kódbázisban látható megoldásomnál arra törekedtem, hogy könnyű legyen esettanulmányokat elvégezni. A **tbaihw.ipynb** demostrálja az eszköz használatát. A kódbázis klónozása és a requirements.txt alapján a megfelelő környezet kialakítása után futtatható. (És ezzel létrejön a Task4 táblázat LD08-as sora.) Feltétel még, hogy a TensorBoard el legyen indítva a tbaihw könyvtában.

Amint azt az osztályok, függvények kommentjeiben részletezem, a VAEC (Variational AutoEncoder and Classifier) osztályból készített példány becsomagolja a tanító-adatokat, és a legfontosabb adatokat (pl config), illetve létrehozza és tartja az encoder, decoder és classifier_head komponenseket. Ezek különféle stratégiával történő betanítása a fit_all_parts(), fit_autoencoder(), fit_classifier_head() és fit_mlp_classifier() tagfüggvényekkel lehetséges:
- fit_all_parts(): Az összes komponenset egyszerre tanítjuk.
- fit_autoencoder(): Csak az encoder-t és decoder-t tanítjuk VAE tanítással.
- fit_classifier_head(): Csak a classifier_head-et tanítjuk.
- fit_mlp_classifier(): Egybe tanítjuk az encodert és a classifier_head-et. Ekkor a kapott modell tisztán MLP classifiernek tekinthető.

A betanítást követően a compile-olás nélkül létrehozott self.mlp_classifier és self.mlp_classifier_with_z használhatóak mint tisztán mlp classifier, illetve olyan classifier, amelynél az autoencoder neck layere is az outputra van kötve (a kódban z).

A tanításokat a VAEC példányon belül készített VAEC_Trainer példányok végzik. Fontos észrevenni, hogy ezek ugyanazokat a decoder, encoder és classifier_head komponenseket tartják, mint a VAEC_Trainereket tartó VAEC példány. Ezek feladata tehát az, hogy a tanítás során szabályozzák, hogy a total_loss-nak mely lossok legyenek a részei, továbbá mely komponensek legyenek taníthatóak, illetve nem taníthatóak.

## Task2
Sajnos ez a részfeladat már nem fért bele az időmbe.

## Task3
A megoldásommal kétféle classifiert készíthetünk, egy egyszerű full-mlp classifiert és egy olyat, amelynek encoder része variational autoencoder-ként lett tanítva, így a neck-nél a latent_dim dimenzióba történő vetítés az eredeti output jelentősen dimenzió redukált reprezentációja. Azért a VAE tanítással tanított encoderrel ellátott MLP classifiert választottam, mert a VAE tanítás során a neck-ben (ahol az encoder és decoder találkozik, itt a dimenzió latent_dim nagyságú) úgy reprezentáljuk a tanítóadatot, hogy minden dimenzió szerint (minden neuron kimenetet tekintve) az eloszlás normál eloszláshoz közel essen. Ezt a tulajdonságot aztán fel tudom használni arra, hogy ismeretlen eredetű adatcsomag esetén is tudjak valamit mondani arra, hogy az tekinthető-e az eredeti tanítóadat által reprezentált eloszlás mintavételezésének. (Lásd alább kifejtve.)

## Task4
A különféle tanításokkal elért test accuracy eredményeket tartalmazza az alábbi táblázat:

|      |fit_ae_then_cl_head|fit_all_ae_then_cl_head|fit_all_parts|fit_mlp_classifier|
|---   |---                |---                    |---          |---               |
| LD16 |       ~92%        |         ~92%          |    ~95%     |      ~95%        |
| LD08 |       ~89%        |        ~90.5%         |    ~95%     |      ~95%        |
| LD04 |      ~84.5%       |         ~84%          |   ~94.5%    |      ~96%        |
| LD02 |       ~75%        |         ~76%          |    ~93%     |      ~95%        |

Ami a fit_mlp_classifier tanításokat illeti, el is vártuk, hogy lényegében azonos és nagy értéket adjanak az így előállt modellek.
A fit_all_parts tanítással előállt modellek alig adnak gyengébb eredményeket. Ugyanakkor, sajnos ezen modellek esetében sejtelmünk sincsen, mennyire megbízható a **z** kimenet annak becslésére, hogy a bemenetre hihetően a tanító adat eloszlásából mintavételzett adat került-e? (Lásd lejjebb a kifejtést.)
A fit_ae_then_cl_head és fit_all_ae_then_cl_head tanítások nagyon hasonló performanszot eredményeztek. Az utóbbi esetben alkalmazott fit_all_parts előtanítás nem különösebben segít. Ezeknél az LD16, LD08, L04, LD02 modellek közül nehéz választani. Minél kisebb a latent_dim (LD), annál "érdekesebb" a predikciónk, ugyanakkor a performance cost mindegyik esetében kellemetlenül nagynak tűnik.

## Task5
Annak vizsgálatára, hogy két minta ugyanazon eloszlásból való mintavételezéssel jött-e létre a Kullback Leibler Divergencia kiszámolása szolgálhat egyik megoldásul. Úgy gondolom, hogy erre annyi forrás szolgál, hogy egyedül az lehet érdekes a számunkra, hogy a jelentkező, vagyis ez esetben én, tudja-e hová nyúljon?
Implementált függvény: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html
Ezen felül a Variational Autoencoder tanítása során is azzal biztosítjuk, hogy a latent_dim számú neuront tartalmazú neck-ben a tanítóadat minden dimenzióban normál eloszlásban legyen reprezentálva, hogy a loss részeként minimalizáljuk a kl_lossot.
Az alapvetően ilyen vizsgálatokra kidolgozott tesztek közül pedig a Shapiro-Wilk tesztet választanám. Szintén széleskörűen irodalmazott, és a scipy.stats is tartalmazza egy implementációját: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html.

## Task6
A tömör válaszom a Task 6-nál az lenne, hogy amennyiben szignifikáns dimenzió redukciót érek el az encoderrel, akkor vagy arról van szó, hogy 
- vannak irreleváns paraméterek,
- vagy a modell enkóder része "érdekes" összefüggések megtanulása révén tudta a dimenzió redukciót ilyen - remélhetőleg - markáns mértékben megoldani.
Az irreleváns paraméterek megállapítására csak brute force megoldás ötletem van. Vennék egy megfelelően egyszerű, tisztán MLP classifiert, aminek inputja latent_dim méretű. Ezt a classifiert tömegesen betanítanám az összesen 30 input paraméteremből minden lehetséges latent_dim számú paraméter szubszetet tanítóadatnak használva. Ez sajnos 30 alatt a latent_dim számú kombinációt jelent, ami pl latent_dim=4-re 27405, így a módszer már itt meghal. A házi feladathoz meghatározott adatcsomag esetében amúgy szerepel ez a mondat: "best predictive accuracy obtained using one separating plane in the 3-D space of Worst Area, Worst Smoothness and Mean Texture". latent_dim=3 esetben a kombinációk száma 4060, ezt egy nagyon egyszerű MLP classifierrel éppen végig lehet számolni. A konklúziót pedig az alapján hoznám meg, hogy van-e olyan paraméter szubszet, amivel csekély performance cost árán is be lehet tanítani a classifiert?

A "feature selection techniques in machine learning" kereső kifejezéssel amúgy számos forrás található, amelyek segíthetnek megállapítani az irreleváns paramétereket. Ezek megemésztése már nem fért bele az időmbe, éles helyzetben választanék egy megfelelő módszert az itt sorjázó irodalomból, pl: https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/

## Task7
És eljutottunk végül a fő kérdésig, amelynek megválaszolhatóságát szem előtt tartva választottam ki a variational autoencoder-es megoldásomat. Amennyiben ismeretlen minőségű adatot kapunk targetek nélkül, akkor az mlp_classifier_with_z classifiert használnám, ami szigorúan úgy lett betanítva, hogy először fit_autoencoder()-rel betanítjuk a VAE-t, majd ezt követően a fit_classifier_head()-del betanítjuk a classifier head-et is. Ezután egy adott input prediktálása során a latent_dim méretű "z" output vektor minden egyes elemére T-teszttel megmondjuk, hogy milyen konfidencia érték mellett tekinthető az értéke egy normál-eloszlásból történő mintavételezésnek. Így egy konfidencia-vektort kapunk, aminek értelmezése még kérdéseket vet fel. Ugyanezt a tesztet elvégezhetjük a vektor hosszértékével is, ekkor egyetlen konfidencia értéket kapunk. Ezzel tehát azt becsülnénk meg inputonként (egy adatsor 30 bemenő értékkel), hogy milyen konfidenciával jelenthetjük ki, hogy legalábbis a neck-beli reprezentációs térben az eredeti tanító adatnak megfelelő normál eloszlás mintavéltelének tekinthető-e az adott input?

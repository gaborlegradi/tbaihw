# Házi feladat a Turbine részére - 1st part

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
A megoldásommal kétféle classifiert készíthetünk, egy egyszerű full-mlp classifiert és egy olyat, amelynek encoder része variational autoencoder-ként lett tanítva, így a neck-nél a latent_dim dimenzióba történő vetítés az eredeti output jelentősen dimenzió redukált reprezentációja. Azért a VAE tanítással tanított encoderrel ellátott MLP classifiert választottam, mert a VAE tanítás során a neck-ben (ahol az encoder és decoder találkozik, itt a dimenzió latent_dim nagyságú) úgy reprezentáljuk a tanítóadatot, hogy minden dimenzió szerint (minden neuron kimenetet tekintve) az eloszlás normál eloszláshoz közel essen. Ezt a tulajdonságot aztán fel tudom használni arra, hogy ismeretlen eredetű adatcsomag esetén is tudjak valamit mondani arra, hogy az tekinthető-e az eredeti tanítóadat által reprezentált eloszlás mintavételezésének.

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

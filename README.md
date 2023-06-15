# Házi feladat Turbine részére - 1st part

## A választott megoldás

A házi feladat első részének kiírásában a Task6 és Task7 mozgatta meg legjobban a fantáziámat. Úgy ítéltem meg, hogy ezen kérdésekre akkor tudok válaszolni, ha a modellemnek része egy variational autoenkóder. Az alapötletem az volt, hogy betanítok egy VAE-t, majd ennek az encoder részére ráhelyezve egy MLP classifier head-et kapok egy tisztán MLP classifiert.

Egy megoldást gyorsan össze is drótoztam egy Keras tutorialból kiindulva és teszteltem MNIST adaton. Itt már a latent_dim = 1 esetre is ~60% körüli accuracy-t kaptam, a latent_dim 2, illetve 3 értékénél 80% és 90% fölötti accuracy tűnt elérhetőenk. Ezt ígéretesnek ítélve rátértem a Wisconsin Breast Cancer Dataset-re készítendő megoldásomra.

A kódbázisban látható megoldásomnál arra törekedtem, hogy könnyű legyen esettanulmányokat elvégezni. A tbaihw.ipynb demostrálja az eszköz használatát. A kódbázis klónozása és a requirements.txt alapján a megfelelő környezet kialakítása után futtatható. Feltétel még, hogy a TensorBoard el legyen indítva a tbaihw könyvtában.

Amint azt az osztályok, függvények kommentjeiben részletezem, a VAEC (Variational AutoEncoder and Classifier) osztályból készített példány becsomagolja a tanító-adatokat, és a legfontosabb adatokat (pl config), illetve létrehozza és tartja az encoder, decoder és classifier_head komponenseket. Ezek különféle stratégiával történő betanítása a fit_all_parts(), fit_autoencoder(), fit_classifier_head() és fit_mlp_classifier() tagfüggvényekkel lehetséges. A betanítást követően a compile-olás nélkül létrehozott self.mlp_classifier és self.mlp_classifier_with_z használhatóak mint tisztán mlp classifier, illetve olyan classifier, amelynél az autoencoder neck része is az outputra van kötve (z).

A tanításokat a VAEC_Trainer példányok végzik. Fontos észrevenni, hogy ezek ugyanazokat a decoder, encoder és classifier_head komponenseket tartják, mint a VAEC_Trainereket tartó VAEC példány. Ezek feladata tehát az, hogy a tanítás során szabályozzák, hogy a total_loss-nak mely lossok legyenek a részei, ileltve mely komponensek legyenek taníthatóak, ileltve nem taníthatóak.

Az alábbiakban megyek végig a feladat taskjain:

## Task1
Teljesítve.

## Task2
Sajnos ez a részfeladat már nem fért bele az időmba.

## Task3
A megoldásommal kétféle classifiert készíthetünk, egy egyszerű full-mlp classifiert és egy olyat, amelynek encoder része variational autoencoder-ként lett tanítva, így a neck-nél a latent_dim dimenzióba történő vetítés az eredeti output jelentősen dimenzió redukált 

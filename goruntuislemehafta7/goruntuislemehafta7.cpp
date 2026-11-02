#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <stdexcept>

using namespace cv;
using namespace std;

// 1. Sinus sinyali.
vector<double> sinusOlustur(
    int N, double genlik, double fazDerece,
    double frekans, double orneklemeFrekansi)
{
    vector<double> veri(N);
    double faz = fazDerece * CV_PI / 180.0;

    for (int n = 0; n < N; n++)
    {
        double t = n / orneklemeFrekansi;

        veri[n] = genlik * sin(
            2.0 * CV_PI * frekans * t + faz
        );
    }

    return veri;
}

// 2. Onceki maddeyle ayni gurultu: ayni tohum ve oran.
vector<double> gurultuEkle(
    const vector<double>& veri,
    double oran)
{
    double tepe = 0.0;

    for (double deger : veri)
        if (abs(deger) > tepe)
            tepe = abs(deger);

    mt19937 uretec(42);
    uniform_real_distribution<double> dagilim(-1.0, 1.0);

    vector<double> sonuc = veri;

    for (size_t n = 0; n < veri.size(); n++)
        sonuc[n] += dagilim(uretec) * tepe * oran;

    return sonuc;
}

// Sinirlarda ilk/son ornek tekrar edilir.
int sinirIndeksi(int indeks, int N)
{
    if (indeks < 0)
        return 0;

    if (indeks >= N)
        return N - 1;

    return indeks;
}

// 3. Gauss filtresini kendimiz uyguluyoruz.
vector<double> gaussFiltre(
    const vector<double>& veri,
    int pencere,
    double sigma)
{
    if (veri.empty() || pencere < 1 ||
        pencere % 2 == 0 || sigma <= 0)
    {
        throw invalid_argument("Gauss parametreleri gecersiz.");
    }

    int yari = pencere / 2;
    vector<double> cekirdek(pencere);

    double toplam = 0.0;

    // Gauss agirliklarini olustur.
    for (int k = -yari; k <= yari; k++)
    {
        double agirlik = exp(
            -(static_cast<double>(k) * k) /
            (2.0 * sigma * sigma)
        );

        cekirdek[k + yari] = agirlik;
        toplam += agirlik;
    }

    // Agirliklarin toplami 1 olsun.
    for (double& agirlik : cekirdek)
        agirlik /= toplam;

    vector<double> sonuc(veri.size(), 0.0);
    int N = static_cast<int>(veri.size());

    for (int n = 0; n < N; n++)
    {
        for (int k = -yari; k <= yari; k++)
        {
            int komsu = sinirIndeksi(n + k, N);

            sonuc[n] +=
                veri[komsu] * cekirdek[k + yari];
        }
    }

    return sonuc;
}

// 4. Medyan filtresini kendimiz uyguluyoruz.
vector<double> medyanFiltre(
    const vector<double>& veri,
    int pencere)
{
    if (veri.empty() || pencere < 1 || pencere % 2 == 0)
        throw invalid_argument("Medyan penceresi pozitif ve tek olmali.");

    vector<double> sonuc(veri.size());
    vector<double> komsular(pencere);

    int yari = pencere / 2;
    int N = static_cast<int>(veri.size());

    for (int n = 0; n < N; n++)
    {
        for (int k = -yari; k <= yari; k++)
        {
            int komsu = sinirIndeksi(n + k, N);
            komsular[k + yari] = veri[komsu];
        }

        sort(komsular.begin(), komsular.end());
        sonuc[n] = komsular[yari];
    }

    return sonuc;
}

// Temiz sinyale gore ortalama karesel hata.
double mseHesapla(
    const vector<double>& temiz,
    const vector<double>& sonuc)
{
    double toplam = 0.0;

    for (size_t n = 0; n < temiz.size(); n++)
    {
        double fark = temiz[n] - sonuc[n];
        toplam += fark * fark;
    }

    return toplam / temiz.size();
}

// Temiz referansi ve incelenen sinyali ayni eksende ciz.
Mat grafikCiz(
    const vector<double>& temiz,
    const vector<double>& sonuc,
    double fs,
    double ySiniri,
    const string& baslik)
{
    const int W = 1100;
    const int H = 500;

    const int sol = 85;
    const int ust = 85;
    const int grafikW = 980;
    const int grafikH = 340;

    Mat grafik(H, W, CV_8UC3, Scalar(255, 255, 255));

    for (int i = 0; i <= 4; i++)
    {
        int y = ust + cvRound(i * grafikH / 4.0);
        double deger = ySiniri - i * ySiniri / 2.0;

        line(
            grafik, Point(sol, y), Point(sol + grafikW, y),
            Scalar(220, 220, 220)
        );

        putText(
            grafik, format("%.2f", deger),
            Point(8, y + 5), FONT_HERSHEY_SIMPLEX,
            0.45, Scalar(0, 0, 0), 1
        );
    }

    double sonZaman = (temiz.size() - 1) / fs;

    for (int i = 0; i <= 5; i++)
    {
        int x = sol + cvRound(i * grafikW / 5.0);

        line(
            grafik, Point(x, ust), Point(x, ust + grafikH),
            Scalar(220, 220, 220)
        );

        putText(
            grafik, format("%.2f", i * sonZaman / 5.0),
            Point(x - 18, ust + grafikH + 25),
            FONT_HERSHEY_SIMPLEX, 0.45,
            Scalar(0, 0, 0), 1
        );
    }

    rectangle(
        grafik,
        Point(sol, ust),
        Point(sol + grafikW, ust + grafikH),
        Scalar(0, 0, 0)
    );

    auto nokta = [&](size_t n, double deger) -> Point
        {
            return Point(
                sol + cvRound(
                    static_cast<double>(n) /
                    (temiz.size() - 1) * grafikW
                ),
                ust + cvRound(
                    (ySiniri - deger) /
                    (2.0 * ySiniri) * grafikH
                )
            );
        };

    // Once sonuc, sonra temiz referans.
    for (size_t n = 1; n < temiz.size(); n++)
    {
        line(
            grafik,
            nokta(n - 1, sonuc[n - 1]),
            nokta(n, sonuc[n]),
            Scalar(0, 0, 220), 1, LINE_AA
        );
    }

    for (size_t n = 1; n < temiz.size(); n++)
    {
        line(
            grafik,
            nokta(n - 1, temiz[n - 1]),
            nokta(n, temiz[n]),
            Scalar(220, 70, 0), 1, LINE_AA
        );
    }

    putText(
        grafik, baslik, Point(sol, 28),
        FONT_HERSHEY_SIMPLEX, 0.65,
        Scalar(0, 0, 0), 2
    );

    putText(
        grafik, "Mavi: temiz referans",
        Point(sol, 58), FONT_HERSHEY_SIMPLEX,
        0.5, Scalar(220, 70, 0), 1
    );

    putText(
        grafik, "Kirmizi: incelenen sinyal",
        Point(sol + 310, 58), FONT_HERSHEY_SIMPLEX,
        0.5, Scalar(0, 0, 220), 1
    );

    putText(
        grafik, "Genlik", Point(8, 65),
        FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 0), 1
    );

    putText(
        grafik, "Zaman (s)", Point(510, 485),
        FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 0), 1
    );

    return grafik;
}

int main()
{
    try
    {
        const int N = 512;
        const double fs = 512.0;

        // Onceki maddeyle ayni sinus ve gurultu.
        vector<double> temiz =
            sinusOlustur(N, 2.0, 30.0, 5.0, fs);

        vector<double> gurultulu =
            gurultuEkle(temiz, 0.30);

        // Ayri ayri filtrele.
        vector<double> gauss =
            gaussFiltre(gurultulu, 9, 1.5);

        vector<double> medyan =
            medyanFiltre(gurultulu, 5);

        // Ayrica sirayla Gauss -> Medyan uygula.
        vector<double> sirali =
            medyanFiltre(gauss, 5);

        double hataGurultulu = mseHesapla(temiz, gurultulu);
        double hataGauss = mseHesapla(temiz, gauss);
        double hataMedyan = mseHesapla(temiz, medyan);
        double hataSirali = mseHesapla(temiz, sirali);

        cout << "Temiz sinyale gore MSE degerleri:\n";
        cout << "Gurultulu: " << hataGurultulu << endl;
        cout << "Gauss: " << hataGauss << endl;
        cout << "Medyan: " << hataMedyan << endl;
        cout << "Gauss -> Medyan: " << hataSirali << endl;

        ofstream csv("gorev3_filtre_verileri.csv");

        if (!csv)
            throw runtime_error("CSV dosyasi olusturulamadi.");

        csv << "ornek,zaman_s,temiz,gurultulu,gauss,medyan,gauss_medyan\n";

        for (int n = 0; n < N; n++)
        {
            csv << n << "," << n / fs << ","
                << temiz[n] << "," << gurultulu[n] << ","
                << gauss[n] << "," << medyan[n] << ","
                << sirali[n] << "\n";
        }

        csv.close();

        ofstream rapor("gorev3_hata_raporu.txt");

        if (!rapor)
            throw runtime_error("Rapor dosyasi olusturulamadi.");

        rapor << "Gurultulu MSE: " << hataGurultulu << "\n";
        rapor << "Gauss MSE: " << hataGauss << "\n";
        rapor << "Medyan MSE: " << hataMedyan << "\n";
        rapor << "Gauss -> Medyan MSE: " << hataSirali << "\n";

        rapor.close();

        // Tum grafiklerde ayni genlik olcegi.
        const double ySiniri = 3.0;

        Mat g1 = grafikCiz(
            temiz, gurultulu, fs, ySiniri,
            "Gurultulu Sinyal"
        );

        Mat g2 = grafikCiz(
            temiz, gauss, fs, ySiniri,
            "Gauss Filtresi - Pencere: 9, Sigma: 1.5"
        );

        Mat g3 = grafikCiz(
            temiz, medyan, fs, ySiniri,
            "Medyan Filtresi - Pencere: 5"
        );

        Mat g4 = grafikCiz(
            temiz, sirali, fs, ySiniri,
            "Sirali Filtreleme: Gauss -> Medyan"
        );

        imwrite("gorev3_gurultulu.png", g1);
        imwrite("gorev3_gauss.png", g2);
        imwrite("gorev3_medyan.png", g3);
        imwrite("gorev3_gauss_medyan.png", g4);

        imshow("Gurultulu", g1);
        imshow("Gauss", g2);
        imshow("Medyan", g3);
        imshow("Gauss -> Medyan", g4);

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}
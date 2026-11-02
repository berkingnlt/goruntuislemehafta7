#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>

using namespace cv;
using namespace std;

// 1. maddede kullandigimiz sinus verisini olustur.
void sinusOlustur(
    vector<double>& veri,
    double genlik,
    double fazDerece,
    double frekans,
    double orneklemeFrekansi)
{
    size_t N = veri.size();

    if (N < 2 || (N & (N - 1)) != 0)
        throw invalid_argument("Vektor uzunlugu 2'nin kuvveti olmali.");

    if (genlik < 0 || orneklemeFrekansi <= 0 ||
        frekans <= 0 || frekans >= orneklemeFrekansi / 2.0)
    {
        throw invalid_argument("Sinus parametreleri gecersiz.");
    }

    double faz = fazDerece * CV_PI / 180.0;

    for (size_t n = 0; n < N; n++)
    {
        double zaman = n / orneklemeFrekansi;

        veri[n] = genlik * sin(
            2.0 * CV_PI * frekans * zaman + faz
        );
    }
}

// Girdi: vektor ve [0,1] araliginda gurultu orani.
// Cikti: gurultu eklenmis yeni vektor.
// Orijinal vektor degistirilmez.
vector<double> gurultuEkle(
    const vector<double>& veri,
    double gurultuOrani)
{
    if (veri.empty())
        throw invalid_argument("Vektor bos olamaz.");

    if (!isfinite(gurultuOrani) ||
        gurultuOrani < 0.0 || gurultuOrani > 1.0)
    {
        throw invalid_argument("Gurultu orani [0,1] araliginda olmali.");
    }

    // Sinyalin tepe genligini vektorden bul.
    double tepeGenlik = 0.0;

    for (double deger : veri)
    {
        double mutlakDeger = abs(deger);

        if (mutlakDeger > tepeGenlik)
            tepeGenlik = mutlakDeger;
    }

    double gurultuSiniri = tepeGenlik * gurultuOrani;

    // Sabit tohum: tekrar calistirinca ayni deney elde edilir.
    mt19937 uretec(42);
    uniform_real_distribution<double> dagilim(-1.0, 1.0);

    vector<double> gurultulu = veri;

    for (size_t n = 0; n < veri.size(); n++)
    {
        double gurultu = dagilim(uretec) * gurultuSiniri;
        gurultulu[n] += gurultu;
    }

    return gurultulu;
}

// Iki grafigin eksenleri aynidir; karsilastirma adildir.
Mat grafikCiz(
    const vector<double>& veri,
    double orneklemeFrekansi,
    double ySiniri,
    const string& baslik,
    Scalar renk)
{
    const int W = 1100;
    const int H = 500;

    const int sol = 85;
    const int sag = 30;
    const int ust = 65;
    const int alt = 70;

    int grafikW = W - sol - sag;
    int grafikH = H - ust - alt;

    Mat grafik(H, W, CV_8UC3, Scalar(255, 255, 255));

    double sonZaman =
        static_cast<double>(veri.size() - 1) /
        orneklemeFrekansi;

    // Genlik izgara cizgileri.
    for (int i = 0; i <= 4; i++)
    {
        int y = ust + cvRound(i * grafikH / 4.0);
        double deger = ySiniri - i * ySiniri / 2.0;

        line(
            grafik,
            Point(sol, y),
            Point(sol + grafikW, y),
            Scalar(220, 220, 220)
        );

        putText(
            grafik, format("%.2f", deger),
            Point(8, y + 5),
            FONT_HERSHEY_SIMPLEX, 0.45,
            Scalar(0, 0, 0), 1
        );
    }

    // Zaman izgara cizgileri.
    for (int i = 0; i <= 5; i++)
    {
        int x = sol + cvRound(i * grafikW / 5.0);

        line(
            grafik,
            Point(x, ust),
            Point(x, ust + grafikH),
            Scalar(220, 220, 220)
        );

        putText(
            grafik, format("%.2f", i * sonZaman / 5.0),
            Point(x - 18, ust + grafikH + 25),
            FONT_HERSHEY_SIMPLEX, 0.45,
            Scalar(0, 0, 0), 1
        );
    }

    int sifirY = ust + cvRound(grafikH / 2.0);

    line(
        grafik,
        Point(sol, sifirY),
        Point(sol + grafikW, sifirY),
        Scalar(100, 100, 100)
    );

    rectangle(
        grafik,
        Point(sol, ust),
        Point(sol + grafikW, ust + grafikH),
        Scalar(0, 0, 0)
    );

    auto nokta = [&](size_t n) -> Point
        {
            int x = sol + cvRound(
                static_cast<double>(n) /
                (veri.size() - 1) * grafikW
            );

            int y = ust + cvRound(
                (ySiniri - veri[n]) /
                (2.0 * ySiniri) * grafikH
            );

            return Point(x, y);
        };

    for (size_t n = 1; n < veri.size(); n++)
    {
        line(
            grafik,
            nokta(n - 1),
            nokta(n),
            renk, 2, LINE_AA
        );
    }

    putText(
        grafik, baslik, Point(sol, 35),
        FONT_HERSHEY_SIMPLEX, 0.7,
        Scalar(0, 0, 0), 2
    );

    putText(
        grafik, "Genlik", Point(8, 35),
        FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 0), 1
    );

    putText(
        grafik, "Zaman (s)", Point(W / 2 - 40, H - 15),
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

        double genlik = 2.0;
        double fazDerece = 30.0;
        double frekans = 5.0;
        double orneklemeFrekansi = 512.0;

        // 0: gurultu yok, 1: tepe genliginin ±%100'u.
        double gurultuOrani = 0.30;

        vector<double> temiz(N);

        sinusOlustur(
            temiz,
            genlik,
            fazDerece,
            frekans,
            orneklemeFrekansi
        );

        vector<double> gurultulu =
            gurultuEkle(temiz, gurultuOrani);

        ofstream dosya("gorev2_gurultulu_veriler.csv");

        if (!dosya)
            throw runtime_error("CSV dosyasi olusturulamadi.");

        dosya << "ornek,zaman_s,temiz,eklenen_gurultu,gurultulu\n";

        double hataKareleriToplami = 0.0;

        for (int n = 0; n < N; n++)
        {
            double gurultu = gurultulu[n] - temiz[n];

            hataKareleriToplami += gurultu * gurultu;

            dosya << n << ","
                << n / orneklemeFrekansi << ","
                << temiz[n] << ","
                << gurultu << ","
                << gurultulu[n] << "\n";
        }

        dosya.close();

        cout << "Ornek sayisi: " << N << endl;
        cout << "Gurultu orani: " << gurultuOrani << endl;

        cout << "Ortalama karesel hata (MSE): "
            << hataKareleriToplami / N << endl;

        // Her iki grafik icin ortak genlik olcegi.
        double ySiniri = genlik * (1.0 + gurultuOrani) * 1.1;

        if (ySiniri <= 0)
            ySiniri = 1.0;

        Mat temizGrafik = grafikCiz(
            temiz, orneklemeFrekansi, ySiniri,
            "Temiz Sinus Sinyali",
            Scalar(220, 70, 0)
        );

        Mat gurultuluGrafik = grafikCiz(
            gurultulu, orneklemeFrekansi, ySiniri,
            "Gurultulu Sinus - Oran: " + format("%.2f", gurultuOrani),
            Scalar(0, 0, 220)
        );

        imwrite("gorev2_temiz_sinyal.png", temizGrafik);
        imwrite("gorev2_gurultulu_sinyal.png", gurultuluGrafik);

        imshow("Temiz Sinyal", temizGrafik);
        imshow("Gurultulu Sinyal", gurultuluGrafik);

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}
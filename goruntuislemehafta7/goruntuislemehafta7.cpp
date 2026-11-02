#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>

using namespace cv;
using namespace std;

// Sinus olustur.
vector<double> sinusOlustur(
    int N, double genlik, double frekans, double fs)
{
    vector<double> veri(N);

    for (int n = 0; n < N; n++)
    {
        veri[n] = genlik * sin(
            2.0 * CV_PI * frekans * n / fs
        );
    }

    return veri;
}

// Her sinyal icin farkli tohumla bagimsiz gurultu.
vector<double> gurultuEkle(
    const vector<double>& veri,
    double oran,
    unsigned int tohum)
{
    if (oran < 0.0 || oran > 1.0)
        throw invalid_argument("Gurultu orani [0,1] olmali.");

    double tepe = 0.0;

    for (double deger : veri)
        if (abs(deger) > tepe)
            tepe = abs(deger);

    mt19937 uretec(tohum);
    uniform_real_distribution<double> dagilim(-1.0, 1.0);

    vector<double> sonuc = veri;

    for (size_t n = 0; n < veri.size(); n++)
        sonuc[n] += dagilim(uretec) * tepe * oran;

    return sonuc;
}

// Hazir dft/FFT kullanmadan manuel DFT.
vector<complex<double>> kendiDFT(const vector<double>& veri)
{
    size_t boyut = veri.size();

    if (boyut < 2 || (boyut & (boyut - 1)) != 0)
        throw invalid_argument("Uzunluk 2'nin kuvveti olmali.");

    int N = static_cast<int>(boyut);
    vector<complex<double>> harmonikler(N);

    for (int k = 0; k < N; k++)
    {
        double gercek = 0.0;
        double sanal = 0.0;

        for (int n = 0; n < N; n++)
        {
            double aci = 2.0 * CV_PI * k * n / N;

            gercek += veri[n] * cos(aci);
            sanal -= veri[n] * sin(aci);
        }

        harmonikler[k] = complex<double>(gercek, sanal);
    }

    return harmonikler;
}

// Karmasik harmonik vektorunden tek tarafli genlik spektrumu.
vector<double> genlikSpektrumu(
    const vector<complex<double>>& harmonikler)
{
    int N = static_cast<int>(harmonikler.size());
    vector<double> genlikler(N / 2 + 1);

    for (int k = 0; k <= N / 2; k++)
    {
        genlikler[k] = abs(harmonikler[k]) / N;

        if (k != 0 && k != N / 2)
            genlikler[k] *= 2.0;
    }

    return genlikler;
}

// Hem zaman grafigi hem frekans grafigi icin ortak cizici.
// xAdimi: zaman icin 1/fs, frekans icin fs/N.
Mat grafikCiz(
    const vector<double>& veri,
    double xAdimi,
    double xMax,
    double yMin,
    double yMax,
    const string& baslik,
    const string& xEtiketi,
    bool spektrum,
    Scalar renk)
{
    const int W = 1100;
    const int H = 500;
    const int sol = 85;
    const int ust = 65;
    const int grafikW = 980;
    const int grafikH = 350;

    Mat grafik(H, W, CV_8UC3, Scalar(255, 255, 255));

    auto nokta = [&](double x, double y) -> Point
        {
            return Point(
                sol + cvRound(x / xMax * grafikW),
                ust + cvRound(
                    (yMax - y) / (yMax - yMin) * grafikH
                )
            );
        };

    for (int i = 0; i <= 4; i++)
    {
        double yDegeri = yMax - i * (yMax - yMin) / 4.0;
        int y = nokta(0, yDegeri).y;

        line(
            grafik, Point(sol, y),
            Point(sol + grafikW, y),
            Scalar(220, 220, 220)
        );

        putText(
            grafik, format("%.2f", yDegeri),
            Point(5, y + 5),
            FONT_HERSHEY_SIMPLEX, 0.45,
            Scalar(0, 0, 0), 1
        );
    }

    for (int i = 0; i <= 5; i++)
    {
        double xDegeri = i * xMax / 5.0;
        int x = nokta(xDegeri, 0).x;

        line(
            grafik, Point(x, ust),
            Point(x, ust + grafikH),
            Scalar(220, 220, 220)
        );

        putText(
            grafik, format("%.2f", xDegeri),
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

    int sifirY = nokta(0, 0).y;

    line(
        grafik,
        Point(sol, sifirY),
        Point(sol + grafikW, sifirY),
        Scalar(120, 120, 120)
    );

    for (size_t n = 0; n < veri.size(); n++)
    {
        double x = n * xAdimi;

        if (x > xMax)
            break;

        Point p = nokta(x, veri[n]);

        if (spektrum)
        {
            line(grafik, Point(p.x, sifirY), p, renk, 2);
            circle(grafik, p, 2, renk, FILLED);
        }
        else if (n > 0)
        {
            line(
                grafik,
                nokta((n - 1) * xAdimi, veri[n - 1]),
                p, renk, 1, LINE_AA
            );
        }
    }

    putText(
        grafik, baslik, Point(sol, 32),
        FONT_HERSHEY_SIMPLEX, 0.6,
        Scalar(0, 0, 0), 2
    );

    putText(
        grafik, "Genlik", Point(8, 35),
        FONT_HERSHEY_SIMPLEX, 0.5,
        Scalar(0, 0, 0), 1
    );

    putText(
        grafik, xEtiketi, Point(490, 480),
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
        const double gurultuOrani = 0.30;

        // 1. Iki temiz sinyal.
        vector<double> temiz1 = sinusOlustur(N, 12.0, 50.0, fs);
        vector<double> temiz2 = sinusOlustur(N, 5.0, 30.0, fs);

        // 2. Ayri gurultuler ekle.
        vector<double> sinyal1 =
            gurultuEkle(temiz1, gurultuOrani, 42);

        vector<double> sinyal2 =
            gurultuEkle(temiz2, gurultuOrani, 123);

        // 3. Gurultulu sinyallerin DFT analizlerini ayri yap.
        vector<complex<double>> DFT1 = kendiDFT(sinyal1);
        vector<complex<double>> DFT2 = kendiDFT(sinyal2);

        // 4. Iki gurultulu sinyali ornek ornek topla.
        vector<double> toplam(N);

        for (int n = 0; n < N; n++)
            toplam[n] = sinyal1[n] + sinyal2[n];

        // 5. Toplam sinyalin DFT analizini yap.
        vector<complex<double>> DFTToplam = kendiDFT(toplam);

        vector<double> spektrum1 = genlikSpektrumu(DFT1);
        vector<double> spektrum2 = genlikSpektrumu(DFT2);
        vector<double> spektrumToplam = genlikSpektrumu(DFTToplam);

        // DFT dogrusallik kontrolu:
        // DFT(x1+x2) = DFT(x1) + DFT(x2)
        double enBuyukFark = 0.0;

        for (int k = 0; k < N; k++)
        {
            double fark = abs(
                DFTToplam[k] - (DFT1[k] + DFT2[k])
            );

            if (fark > enBuyukFark)
                enBuyukFark = fark;
        }

        int k30 = cvRound(30.0 * N / fs);
        int k50 = cvRound(50.0 * N / fs);

        cout << "Frekans cozunurlugu: " << fs / N << " Hz\n";
        cout << "Gurultu orani: " << gurultuOrani << "\n\n";

        cout << "Sinyal 1 - 50 Hz genligi: "
            << spektrum1[k50] << endl;

        cout << "Sinyal 2 - 30 Hz genligi: "
            << spektrum2[k30] << endl;

        cout << "Toplam - 50 Hz genligi: "
            << spektrumToplam[k50] << endl;

        cout << "Toplam - 30 Hz genligi: "
            << spektrumToplam[k30] << endl;

        cout << "DFT dogrusallik kontrolu, maksimum fark: "
            << enBuyukFark << endl;

        // 6. Zaman verilerini kaydet.
        ofstream zamanCSV("gorev5_sinyaller.csv");

        if (!zamanCSV)
            throw runtime_error("Zaman CSV olusturulamadi.");

        zamanCSV << "n,zaman_s,sinyal1,sinyal2,toplam\n";

        for (int n = 0; n < N; n++)
        {
            zamanCSV << n << "," << n / fs << ","
                << sinyal1[n] << "," << sinyal2[n] << ","
                << toplam[n] << "\n";
        }

        zamanCSV.close();

        // 7. Tum karmasik harmonik katsayilarini kaydet.
        ofstream harmonikCSV("gorev5_harmonikler.csv");

        if (!harmonikCSV)
            throw runtime_error("Harmonik CSV olusturulamadi.");

        harmonikCSV
            << "k,frekans_hz,s1_gercek,s1_sanal,"
            << "s2_gercek,s2_sanal,toplam_gercek,toplam_sanal\n";

        for (int k = 0; k < N; k++)
        {
            double f = (k <= N / 2)
                ? k * fs / N
                : (k - N) * fs / N;

            harmonikCSV
                << k << "," << f << ","
                << DFT1[k].real() << "," << DFT1[k].imag() << ","
                << DFT2[k].real() << "," << DFT2[k].imag() << ","
                << DFTToplam[k].real() << ","
                << DFTToplam[k].imag() << "\n";
        }

        harmonikCSV.close();

        // 8. Tam tek tarafli spektrumlari kaydet.
        ofstream spektrumCSV("gorev5_spektrumlar.csv");

        if (!spektrumCSV)
            throw runtime_error("Spektrum CSV olusturulamadi.");

        spektrumCSV << "frekans_hz,sinyal1,sinyal2,toplam\n";

        double maxSpektrum = 0.0;

        for (int k = 0; k <= N / 2; k++)
        {
            spektrumCSV
                << k * fs / N << ","
                << spektrum1[k] << ","
                << spektrum2[k] << ","
                << spektrumToplam[k] << "\n";

            if (spektrum1[k] > maxSpektrum)
                maxSpektrum = spektrum1[k];

            if (spektrum2[k] > maxSpektrum)
                maxSpektrum = spektrum2[k];

            if (spektrumToplam[k] > maxSpektrum)
                maxSpektrum = spektrumToplam[k];
        }

        spektrumCSV.close();

        // Ortak eksen olcekleri.
        double maxZaman = 0.0;

        for (int n = 0; n < N; n++)
        {
            if (abs(sinyal1[n]) > maxZaman)
                maxZaman = abs(sinyal1[n]);

            if (abs(sinyal2[n]) > maxZaman)
                maxZaman = abs(sinyal2[n]);

            if (abs(toplam[n]) > maxZaman)
                maxZaman = abs(toplam[n]);
        }

        maxZaman *= 1.1;
        maxSpektrum *= 1.1;

        double sonZaman = (N - 1) / fs;

        // 9. Uc sinyalin zaman grafikleri.
        Mat g1 = grafikCiz(
            sinyal1, 1.0 / fs, sonZaman,
            -maxZaman, maxZaman,
            "Sinyal 1: 50 Hz, A=12 + Gurultu",
            "Zaman (s)", false, Scalar(220, 70, 0)
        );

        Mat g2 = grafikCiz(
            sinyal2, 1.0 / fs, sonZaman,
            -maxZaman, maxZaman,
            "Sinyal 2: 30 Hz, A=5 + Gurultu",
            "Zaman (s)", false, Scalar(0, 150, 0)
        );

        Mat g3 = grafikCiz(
            toplam, 1.0 / fs, sonZaman,
            -maxZaman, maxZaman,
            "Toplam Sinyal: Sinyal 1 + Sinyal 2",
            "Zaman (s)", false, Scalar(0, 0, 220)
        );

        // Grafikler 0-100 Hz'e yakinlastirilir.
        // CSV'de 0-256 Hz araliginin tamami bulunur.
        Mat g4 = grafikCiz(
            spektrum1, fs / N, 100.0,
            0, maxSpektrum,
            "Sinyal 1 - Manuel DFT",
            "Frekans (Hz)", true, Scalar(220, 70, 0)
        );

        Mat g5 = grafikCiz(
            spektrum2, fs / N, 100.0,
            0, maxSpektrum,
            "Sinyal 2 - Manuel DFT",
            "Frekans (Hz)", true, Scalar(0, 150, 0)
        );

        Mat g6 = grafikCiz(
            spektrumToplam, fs / N, 100.0,
            0, maxSpektrum,
            "Toplam Sinyal - Manuel DFT",
            "Frekans (Hz)", true, Scalar(0, 0, 220)
        );

        imwrite("gorev5_sinyal1.png", g1);
        imwrite("gorev5_sinyal2.png", g2);
        imwrite("gorev5_toplam_sinyal.png", g3);
        imwrite("gorev5_dft_sinyal1.png", g4);
        imwrite("gorev5_dft_sinyal2.png", g5);
        imwrite("gorev5_dft_toplam.png", g6);

        imshow("Sinyal 1", g1);
        imshow("Sinyal 2", g2);
        imshow("Toplam Sinyal", g3);
        imshow("DFT - Sinyal 1", g4);
        imshow("DFT - Sinyal 2", g5);
        imshow("DFT - Toplam", g6);

        waitKey(0);
    }
    catch (const exception& hata)
    {
        cerr << "Hata: " << hata.what() << endl;
        return -1;
    }

    return 0;
}